// gpu/vk/dlssg_math.h: NVIDIA DLSS frame generation's camera from the scene's matrices (tests/test_dlssg_math.py)
#include "../gpu/vk/dlssg_math.h"
#include "../gpu/vk/dlss_math.h"
#include "../gpu/vk/fsr3_math.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#define CHECK(name, condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %s at line %d\n", name, __LINE__); return 1; } } while (0)

namespace gm = mkwvk::dlssg_math;

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

// a perspective projection as the game fills it: row major, clip = p * view position, the view looking down -z, clip depth
// -1 (near) to 1 (far), or the other way round (reversed)
static void perspective(float p[16], double fovy, double aspect, double n, double f, bool reversed) {
    for (int i = 0; i < 16; i++) p[i] = 0.0f;
    const double c = 1.0 / std::tan(fovy / 2);
    p[0] = (float)(c / aspect), p[5] = (float)c, p[14] = -1.0f;
    const double zn = reversed ? 1.0 : -1.0, zf = -zn;
    const double p23 = (zn - zf) / (1.0 / n - 1.0 / f), p22 = p23 / n - zn;
    p[10] = (float)p22, p[11] = (float)p23;
}

// a view (world to camera), row major with column vectors: a turn about y by `yaw`, then the camera at `eye`
static void view(float v[16], double yaw, const double eye[3]) {
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double r[3][3] = {{c, 0, -s}, {0, 1, 0}, {s, 0, c}};   // (rows: the camera's right, up and backward in the world)
    for (int i = 0; i < 16; i++) v[i] = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) v[i * 4 + j] = (float)r[i][j];
        v[i * 4 + 3] = (float)-(r[i][0] * eye[0] + r[i][1] * eye[1] + r[i][2] * eye[2]);
    }
    v[15] = 1.0f;
}

static void mul(const float a[16], const float b[16], float out[16]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double s = 0;
            for (int k = 0; k < 4; k++) s += (double)a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = (float)s;
        }
}

// a row vector times a matrix as NGX takes one: out = v * m
static void row_times(const double v[4], const float m[16], double out[4]) {
    for (int c = 0; c < 4; c++) {
        out[c] = 0;
        for (int r = 0; r < 4; r++) out[c] += v[r] * (double)m[r * 4 + c];
    }
}

// a column-vector matrix times a position: out = m * v
static void times(const float m[16], const double v[4], double out[4]) {
    for (int r = 0; r < 4; r++) {
        out[r] = 0;
        for (int c = 0; c < 4; c++) out[r] += (double)m[r * 4 + c] * v[c];
    }
}

static int run(bool reversed) {
    const double fovy = 45.0 * 3.14159265358979 / 180.0, aspect = 16.0 / 9.0, n = 0.1, f = 1000.5;
    float p[16], v0[16], v1[16], vp0[16], vp1[16];
    perspective(p, fovy, aspect, n, f, reversed);
    const double eye0[3] = {3.0, 1.5, -7.0}, eye1[3] = {3.4, 1.5, -6.2};
    view(v0, 0.30, eye0);   // (the frame before)
    view(v1, 0.37, eye1);   // (this frame)
    mul(p, v0, vp0);
    mul(p, v1, vp1);
    // (the depth copy is flipped exactly when the game's is not reversed: fsr3_math.h's flip_depth)
    mkwvk::fsr3_math::Camera fc;
    CHECK("the projection is one FSR's camera reads", mkwvk::fsr3_math::camera_from_projection(p, fc));
    CHECK("flip_depth is the game's depth not being reversed", fc.flip_depth == !reversed);
    gm::Camera cam;
    CHECK("camera", gm::camera(p, vp1, vp0, fc.flip_depth, cam));

    // a point of the world in front of both cameras
    const double world[4] = {-0.5, 0.8, -15.0, 1.0};
    double view_pos[4], clip_gl[4], prev_gl[4];
    times(v1, world, view_pos);
    times(vp1, world, clip_gl);
    times(vp0, world, prev_gl);
    CHECK("the point is in front of the camera", clip_gl[3] > 0 && prev_gl[3] > 0);
    // the depth NGX is given for it: the game's stored depth, (ndc z + 1) / 2, flipped when the copy is
    const double game_depth = (clip_gl[2] / clip_gl[3] + 1.0) * 0.5;
    const double given = fc.flip_depth ? 1.0 - game_depth : game_depth;
    CHECK("the depth given is reversed: nearer is larger", given > 0.0 && given < 1.0);
    {   // (1 at the near plane, 0 at the far one)
        const double near_view[4] = {0, 0, -n, 1}, far_view[4] = {0, 0, -f, 1};
        double a[4], b[4];
        row_times(near_view, cam.view_to_clip, a);
        row_times(far_view, cam.view_to_clip, b);
        CHECK("the near plane is depth 1", near(a[2] / a[3], 1.0, 1e-4));
        CHECK("the far plane is depth 0", std::fabs(b[2] / b[3]) < 1e-4);
    }

    // view_to_clip, with the position as a row: x and y as the game's clip space, z / w the depth given
    double clip[4];
    row_times(view_pos, cam.view_to_clip, clip);
    CHECK("view to clip: x", near(clip[0] / clip[3], clip_gl[0] / clip_gl[3], 1e-5));
    CHECK("view to clip: y", near(clip[1] / clip[3], clip_gl[1] / clip_gl[3], 1e-5));
    CHECK("view to clip: z is the depth given", near(clip[2] / clip[3], given, 1e-5));
    CHECK("view to clip: w is the view distance", near(clip[3], -view_pos[2], 1e-5));
    // clip_to_view is its inverse
    double back[4];
    row_times(clip, cam.clip_to_view, back);
    for (int i = 0; i < 3; i++) CHECK("clip to view", near(back[i] / back[3], view_pos[i], 1e-4));

    // clip_to_prev_clip: from this frame's pixel and the depth given, where the point was, and at which depth
    const double ndc[4] = {clip[0] / clip[3], clip[1] / clip[3], given, 1.0};
    double prev[4];
    row_times(ndc, cam.clip_to_prev_clip, prev);
    const double prev_game_depth = (prev_gl[2] / prev_gl[3] + 1.0) * 0.5;
    const double prev_given = fc.flip_depth ? 1.0 - prev_game_depth : prev_game_depth;
    CHECK("clip to previous clip: x", near(prev[0] / prev[3], prev_gl[0] / prev_gl[3], 1e-4));
    CHECK("clip to previous clip: y", near(prev[1] / prev[3], prev_gl[1] / prev_gl[3], 1e-4));
    CHECK("clip to previous clip: the depth given then", near(prev[2] / prev[3], prev_given, 1e-4));
    CHECK("the camera moved: the point is somewhere else", std::fabs(prev[0] / prev[3] - ndc[0]) > 1e-3);
    // prev_clip_to_clip is its inverse
    const double pndc[4] = {prev[0] / prev[3], prev[1] / prev[3], prev[2] / prev[3], 1.0};
    double again[4];
    row_times(pndc, cam.prev_clip_to_clip, again);
    for (int i = 0; i < 3; i++) CHECK("previous clip to clip", near(again[i] / again[3], ndc[i], 1e-4));
    // (and the same answer as the renderer's own reprojection, in the game's clip space: fsr3_prep.comp's motion)
    float reproject[16];
    CHECK("dlss_math's reprojection", mkwvk::dlss_math::reprojection(vp1, vp0, reproject));
    {
        const double c[4] = {clip_gl[0] / clip_gl[3], clip_gl[1] / clip_gl[3], clip_gl[2] / clip_gl[3], 1.0};
        double r[4];
        times(reproject, c, r);
        CHECK("as the motion vectors' reprojection: x", near(r[0] / r[3], prev[0] / prev[3], 1e-4));
        CHECK("as the motion vectors' reprojection: y", near(r[1] / r[3], prev[1] / prev[3], 1e-4));
    }

    // the camera in the world: its position, and forward the way it looks (the view looks down -z)
    for (int i = 0; i < 3; i++) CHECK("the camera's position", near(cam.position[i], eye1[i], 1e-4));
    const double c = std::cos(0.37), s = std::sin(0.37);
    CHECK("right", near(cam.right[0], c, 1e-5) && near(cam.right[2], -s, 1e-5));
    CHECK("up", near(cam.up[1], 1.0, 1e-5));
    CHECK("forward is minus the view's z row", near(cam.forward[0], -s, 1e-5) && near(cam.forward[2], -c, 1e-5));
    {   // (a point one unit along forward is in front: its view z is -1)
        const double ahead[4] = {eye1[0] + cam.forward[0], eye1[1] + cam.forward[1], eye1[2] + cam.forward[2], 1.0};
        double av[4];
        times(v1, ahead, av);
        CHECK("forward points into the picture", near(av[2], -1.0, 1e-4));
    }
    for (int i = 0; i < 16; i++) CHECK("no lens distortion", cam.clip_to_lens_clip[i] == (i % 5 == 0 ? 1.0f : 0.0f));
    return 0;
}

int main() {
    if (run(false)) return 1;   // (the game's depth as shipped: 0 near, the copy flipped)
    if (run(true)) return 1;    // (a reversed projection: the copy as it is)

    // the same camera twice: no motion, and both reprojections the identity
    {
        float p[16], v[16], vp[16];
        perspective(p, 1.0, 16.0 / 9.0, 0.1, 1000.5, false);
        const double eye[3] = {0, 0, 0};
        view(v, 0.0, eye);
        mul(p, v, vp);
        gm::Camera cam;
        CHECK("a still camera", gm::camera(p, vp, vp, true, cam));
        for (int i = 0; i < 16; i++) {
            CHECK("a still camera reprojects to itself", near(cam.clip_to_prev_clip[i], i % 5 == 0 ? 1.0 : 0.0, 1e-5));
            CHECK("and back", near(cam.prev_clip_to_clip[i], i % 5 == 0 ? 1.0 : 0.0, 1e-5));
        }
    }
    // what is not a perspective projection looking down -z, or cannot be inverted, makes no camera
    {
        float p[16], id[16];
        perspective(p, 1.0, 16.0 / 9.0, 0.1, 1000.5, false);
        for (int i = 0; i < 16; i++) id[i] = i % 5 == 0 ? 1.0f : 0.0f;
        gm::Camera cam;
        CHECK("an orthographic projection", !gm::camera(id, id, id, true, cam));
        float zero[16] = {};
        CHECK("a singular view-projection", !gm::camera(p, zero, p, true, cam));
        CHECK("a singular previous one", !gm::camera(p, p, zero, true, cam));
        float bad[16];
        for (int i = 0; i < 16; i++) bad[i] = p[i];
        bad[0] = NAN;
        CHECK("a matrix that is not finite", !gm::camera(p, bad, p, true, cam));
    }
    // motion in render pixels is normalized by the render size; the jitter goes to clip space, y up
    {
        float s[2], j[2];
        gm::motion_scale(1280, 720, s);
        CHECK("motion scale", near(s[0], 1.0 / 1280, 1e-7) && near(s[1], 1.0 / 720, 1e-7));
        gm::motion_scale(0, 0, s);
        CHECK("no size, no scale", s[0] == 0.0f && s[1] == 0.0f);
        const float px[2] = {0.25f, -0.4f};
        gm::jitter_to_clip(px, 1280, 720, j);
        float round[2];
        mkwvk::dlss_math::jitter_to_pixels(j, 1280, 720, round);
        CHECK("the jitter in clip space is dlss_math's, the other way", near(round[0], px[0], 1e-6) && near(round[1], px[1], 1e-6));
        CHECK("clip y is up", j[1] > 0.0f && j[0] > 0.0f);
    }
    std::printf("PASS: DLSS frame generation's camera: row vectors, the depth image's clip space, forward into the picture\n");
    return 0;
}
