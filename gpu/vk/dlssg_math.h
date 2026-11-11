/* (hookshot fork) NVIDIA DLSS frame generation's camera, from the scene's own matrices: plain functions, no Vulkan, no NGX.
   tests/test_dlssg_math.cpp checks them.

   The renderer keeps the scene's matrices as the game fills them (vk_fsr3_inputs.h Frame): row major with column vectors
   (clip = M * position), a view that looks down -z (the projection's w row is 0 0 -1 0), and NVN's clip depth, -1 to 1.
   NGX's frame generation takes another convention on each of those counts, and nothing fails when one is missed: the made
   frames are only wrong.

    - Vectors are rows (position * M): NVIDIA's own helper builds its camera-to-world matrix with the camera's position in
      the last row and composes clipToPrevClip = clipToView * viewToViewPrev * viewToClipPrev, left to right
      (Streamline's sl_matrix_helpers.h, whose constants NGX is handed unchanged). So every matrix is the transpose of ours.
    - Clip depth is the depth image's: clip z / w is the value NGX reads from the depth it is given. The depth here is
      vk_fsr3_inputs.cpp's copy, in 0..1 and reversed (1 at the near plane; `flip_depth`: the copy is 1 - the game's, else
      the game's own was reversed already). The game's clip z is 2 * depth - 1 of the game's depth, so each matrix has
      that change of clip space on its clip side: z' = (w - z) / 2 when the copy is flipped, (w + z) / 2 when it is not.
    - The camera looks along its forward vector: minus the view's third row, the view looking down -z.
    - Motion vectors are normalized ("so the values are in [-1,1] range", nvsdk_ngx_defs_dlssg.h): ours are render pixels,
      so the scale is 1 / render size.
    - The jitter is in clip space (nvsdk_ngx_params_dlssg.h): ours is render pixels, x right and y down. */
#pragma once

#include <cmath>

namespace mkwvk::dlssg_math {

struct Camera {
    // (row vectors, as NGX takes them: each the transpose of the column-vector matrix it is named for)
    float view_to_clip[16]{};        ///< the projection, into the depth image's clip space
    float clip_to_view[16]{};        ///< its inverse
    float clip_to_prev_clip[16]{};   ///< this frame's clip space to the frame before's
    float prev_clip_to_clip[16]{};   ///< and back
    float clip_to_lens_clip[16]{};   ///< identity: no lens distortion here
    float position[3]{}, right[3]{}, up[3]{}, forward[3]{};   ///< the camera in the world
};

namespace detail {

/// m inverted (row major, 4x4) by Gauss-Jordan with partial pivoting, in doubles; false when it is singular or not finite
inline bool invert(const double m[16], double out[16]) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++) {
            a[r][c] = c < 4 ? m[r * 4 + c] : (c - 4 == r ? 1.0 : 0.0);
            if (!std::isfinite(a[r][c])) return false;
        }
    for (int c = 0; c < 4; c++) {
        int p = c;
        for (int r = c + 1; r < 4; r++)
            if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
        if (std::fabs(a[p][c]) < 1e-12) return false;
        if (p != c)
            for (int k = 0; k < 8; k++) {
                const double t = a[p][k];
                a[p][k] = a[c][k], a[c][k] = t;
            }
        const double inv = 1.0 / a[c][c];
        for (int k = 0; k < 8; k++) a[c][k] *= inv;
        for (int r = 0; r < 4; r++) {
            if (r == c || a[r][c] == 0.0) continue;
            const double f = a[r][c];
            for (int k = 0; k < 8; k++) a[r][k] -= f * a[c][k];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) out[r * 4 + c] = a[r][c + 4];
    return true;
}

inline void multiply(const double a[16], const double b[16], double out[16]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double v = 0;
            for (int k = 0; k < 4; k++) v += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = v;
        }
}

/// A column-vector matrix as NGX takes it (row vectors: its transpose), in floats; false when a value is not one
inline bool store(const double m[16], float out[16]) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            const double v = m[c * 4 + r];
            if (!std::isfinite(v) || std::fabs(v) > 3.0e38) return false;
            out[r * 4 + c] = (float)v;
        }
    return true;
}

}  // namespace detail

/// The game's clip space to the depth image's (and back): x, y and w as they are, z' = (w - z) / 2 with the copy flipped
/// (flip_depth), (w + z) / 2 without. Row major, column vectors
inline void clip_change(bool flip_depth, double to[16], double from[16]) {
    for (int i = 0; i < 16; i++) to[i] = from[i] = i % 5 == 0 ? 1.0 : 0.0;
    to[10] = flip_depth ? -0.5 : 0.5, to[11] = 0.5;        // z' = (w -+ z) / 2
    from[10] = flip_depth ? -2.0 : 2.0, from[11] = flip_depth ? 1.0 : -1.0;   // z = w - 2 z', or 2 z' - w
}

/// The camera NGX is given, from the scene's projection, this frame's view-projection and the frame before's (all without
/// their jitter, as the game fills them: row major, column vectors, NVN's clip depth). False when the projection is not
/// a perspective one looking down -z, or a matrix cannot be inverted: no made frame then
inline bool camera(const float projection[16], const float view_projection[16], const float previous_view_projection[16],
                   bool flip_depth, Camera& out) {
    out = Camera{};
    if (projection[12] != 0.0f || projection[13] != 0.0f || projection[14] != -1.0f || projection[15] != 0.0f) return false;
    double p[16], vp[16], pvp[16], to[16], from[16];
    for (int i = 0; i < 16; i++) p[i] = projection[i], vp[i] = view_projection[i], pvp[i] = previous_view_projection[i];
    clip_change(flip_depth, to, from);
    double ip[16], ivp[16], ipvp[16];
    if (!detail::invert(p, ip) || !detail::invert(vp, ivp) || !detail::invert(pvp, ipvp)) return false;
    double t[16], m[16];
    // the projection into the depth image's clip space, and out of it
    detail::multiply(to, p, m);
    if (!detail::store(m, out.view_to_clip)) return false;
    detail::multiply(ip, from, m);
    if (!detail::store(m, out.clip_to_view)) return false;
    // this frame's clip space to the frame before's: previous * inverse(current), between the two depth images' spaces
    detail::multiply(pvp, ivp, t);
    detail::multiply(to, t, m);
    detail::multiply(m, from, t);
    if (!detail::store(t, out.clip_to_prev_clip)) return false;
    detail::multiply(vp, ipvp, t);
    detail::multiply(to, t, m);
    detail::multiply(m, from, t);
    if (!detail::store(t, out.prev_clip_to_clip)) return false;
    for (int i = 0; i < 16; i++) out.clip_to_lens_clip[i] = i % 5 == 0 ? 1.0f : 0.0f;
    // the view (world to camera): inverse(projection) * view-projection. Its rotation's rows are the camera's axes in the
    // world (x right, y up, z backward), and its position is minus the rotation's transpose times the translation
    double v[16];
    detail::multiply(ip, vp, v);
    for (int c = 0; c < 3; c++) {
        out.right[c] = (float)v[0 * 4 + c];
        out.up[c] = (float)v[1 * 4 + c];
        out.forward[c] = (float)-v[2 * 4 + c];
        out.position[c] = (float)-(v[0 * 4 + c] * v[0 * 4 + 3] + v[1 * 4 + c] * v[1 * 4 + 3] + v[2 * 4 + c] * v[2 * 4 + 3]);
    }
    for (int c = 0; c < 3; c++)
        if (!std::isfinite(out.right[c]) || !std::isfinite(out.up[c]) || !std::isfinite(out.forward[c]) ||
            !std::isfinite(out.position[c]))
            return false;
    return true;
}

/// The scale that normalizes motion vectors in render pixels, as NGX takes them
inline void motion_scale(unsigned render_width, unsigned render_height, float out[2]) {
    out[0] = render_width ? 1.0f / (float)render_width : 0.0f;
    out[1] = render_height ? 1.0f / (float)render_height : 0.0f;
}

/// A jitter in render pixels (x right, y down: vk_fsr3_inputs.h Frame) in clip space (y up): dlss_math.h jitter_to_pixels,
/// the other way
inline void jitter_to_clip(const float pixels[2], unsigned render_width, unsigned render_height, float out[2]) {
    out[0] = render_width ? 2.0f * pixels[0] / (float)render_width : 0.0f;
    out[1] = render_height ? -2.0f * pixels[1] / (float)render_height : 0.0f;
}

}  // namespace mkwvk::dlssg_math
