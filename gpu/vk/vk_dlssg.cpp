// (hookshot) DLSS Frame Generation: gpu/vk/vk_dlssg.h says what it is for and where it slots in.
#include "vk_dlssg.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cfloat>
#include <cstring>

#if defined(HOOKSHOT_NGX)
// The shared NGX session (gpu/vk/vk_ngx.h), which is where NVIDIA's own headers are reached from.
// They are not vendored here and must not be: CONTRIBUTING.md's licensing rule.
#include "vk_ngx.h"

#include <nvsdk_ngx_defs_dlssg.h>
#include <nvsdk_ngx_params_dlssg.h>
#endif
#include "vk_work.h"   // (after the others: the commands here go through the recorder, vk_recorder.h; NGX's own, vk_ngx.cpp)

namespace mkwvk::dlssg {
namespace {

/// Whether this build has an NGX SDK to compile the NGX half against. A constant rather than an
/// `#ifdef` at each use, so both arms of every decision below are compiled in every build.
#if defined(HOOKSHOT_NGX)
constexpr bool kHaveSdk = true;
#else
constexpr bool kHaveSdk = false;
#endif

constexpr uint32_t kMaxMade = 3;   ///< (DLSS 4's multi-frame generation: up to three made frames)

/// RX_VK_DLSSG: 0 never (and none of its device extensions asked for), 1 to 3 the most frames it makes between two of the
/// game's, under what NGX allows here; unset: whatever NGX allows. Which sets it makes is the Frame generation and Frame
/// generation method rows' to say (vk_framegen.cpp, framegen_methods.h), read every frame there
int env_limit() {
    static const int env = []() {
        const char* e = getenv("RX_VK_DLSSG");
        return e && *e ? std::clamp(atoi(e), 0, (int)kMaxMade) : -1;
    }();
    return env;
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool in_general = false;   ///< the UNDEFINED -> GENERAL barrier has been recorded
};

struct State {
    Image made[kMaxMade];
    uint32_t width = 0, height = 0, render_width = 0, render_height = 0;
    VkFormat present_format = VK_FORMAT_UNDEFINED;
    uint32_t ready = 0;        ///< how many of `made` exist
    bool first_evaluation = true;
};
State g;
std::atomic<uint32_t> multi_max{1};   ///< NGX's DLSSG.MultiFrameCountMax here (capable)

/// The device idle, for freeing what a frame in flight may still use: the old size's images and feature. As
/// vk_framegen.cpp's ensure_size does at the same moment (every queue idle, the present queue's too)
void idle() {
    std::lock_guard qk(dev.present_mu);
    device_wait_idle_locked();
}

constexpr VkImageSubresourceRange ONE_LEVEL{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

void drop_image(Image& t) {
    if (t.view) vkDestroyImageView(dev.device, t.view, nullptr);
    if (t.image) mkwvk::destroy_image(t.image, t.memory);
    t = Image{};
}

/// A made frame's image: the format the window is presented in, because this is shown as a frame and
/// not read back into the game's own target. Storage because the network writes it, transfer source
/// because the present path blits it to the window.
bool make_frame(Image& t, uint32_t width, uint32_t height, VkFormat format) {
    drop_image(t);
    t.format = format;
    t.width = std::max(1u, width), t.height = std::max(1u, height);
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = t.format;
    ci.extent = {t.width, t.height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!create_image(ci, t.image, t.memory, false)) {
        logf("vk: dlssg: a %ux%u frame image was not made", t.width, t.height);
        return false;
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = t.format;
    vi.subresourceRange = ONE_LEVEL;
    if (vkCreateImageView(dev.device, &vi, nullptr, &t.view) != VK_SUCCESS) {
        logf("vk: dlssg: a frame image's view was not made");
        drop_image(t);
        return false;
    }
    return true;
}

/// UNDEFINED -> GENERAL, once per image. NGX is handed images in GENERAL, as everything else in this
/// renderer is.
void to_general(VkCommandBuffer cmd, Image& t) {
    if (t.in_general || !t.image) return;
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = ONE_LEVEL;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
    t.in_general = true;
}

// ---- the NGX half ----

#if defined(HOOKSHOT_NGX)

NVSDK_NGX_Parameter* params = nullptr;
NVSDK_NGX_Handle* feature = nullptr;

bool ngx_available() {
    if (!ngx::feature_available(NVSDK_NGX_Parameter_FrameGeneration_Available,
                                NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver,
                                NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor,
                                NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor,
                                "Frame Generation"))
        return false;
    if (!params) params = ngx::allocate_params();
    if (!params) return false;
    // (how many frames it makes between two of the game's on this GPU: "If 1 or not set, multiframe is not supported")
    NVSDK_NGX_Parameter* caps = nullptr;
    if (NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&caps)) && caps) {
        unsigned most = 0;
        if (NVSDK_NGX_SUCCEED(ngx::get(caps, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &most)) && most > 1)
            multi_max.store(std::min(most, kMaxMade), std::memory_order_relaxed);
        NVSDK_NGX_VULKAN_DestroyParameters(caps);
    }
    return true;
}

/// The feature, made at the first evaluation after a change of extents. Creation records into a
/// command buffer, which is why it cannot be done in `prepare()`.
bool ngx_feature(VkCommandBuffer cmd) {
    if (feature) return true;
    if (!params) return false;
    // The keys NGX_VK_CREATE_DLSSG sets, set here for the reason vk_dlss.cpp gives: using the
    // inline helper would put NVSDK_NGX_VULKAN_CreateFeature in this object file and hand the link
    // back to NVIDIA's library. The keys themselves are the SDK's own macros, so a wrong one is a
    // compile error.
    ngx::set(params, NVSDK_NGX_Parameter_CreationNodeMask, 1u);
    ngx::set(params, NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
    ngx::set(params, NVSDK_NGX_Parameter_Width, g.width);
    ngx::set(params, NVSDK_NGX_Parameter_Height, g.height);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_BackbufferFormat, (unsigned)g.present_format);
    const NVSDK_NGX_Result res =
        ngx::create_feature(cmd, NVSDK_NGX_Feature_FrameGeneration, params, &feature);
    if (NVSDK_NGX_FAILED(res) || !feature) {
        logf("vk: dlssg: the feature was not created for %ux%u (0x%08x)", g.width, g.height,
             (unsigned)res);
        feature = nullptr;
        return false;
    }
    logf("vk: dlssg: feature ready, %ux%u from %ux%u, %u made frame(s) a frame", g.width, g.height,
         g.render_width, g.render_height, g.ready);
    return true;
}

bool ngx_generate(VkCommandBuffer cmd, const Inputs& in, Image& out) {
    if (!feature || !params) return false;
    NVSDK_NGX_Resource_VK colour =
        ngx::resource(in.color, in.color_image, in.color_format, in.width, in.height, false);
    NVSDK_NGX_Resource_VK depth = ngx::resource(in.depth, in.depth_image, in.depth_format,
                                                in.render_width, in.render_height, false);
    NVSDK_NGX_Resource_VK motion = ngx::resource(in.motion, in.motion_image, in.motion_format,
                                                 in.render_width, in.render_height, false);
    NVSDK_NGX_Resource_VK made =
        ngx::resource(out.view, out.image, out.format, out.width, out.height, true);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_Backbuffer, &colour);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_Depth, &depth);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MVecs, &motion);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_OutputInterpolated, &made);
    // The optional inputs this game does not have: no separate HUD-less colour, no UI layer, no
    // distortion field. They are set to null rather than left from the frame before, because a
    // parameter block is reused and a stale resource is a use-after-free inside the runtime.
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_HUDLess, (void*)nullptr);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_UI, (void*)nullptr);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_UIAlpha, (void*)nullptr);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionField, (void*)nullptr);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_OutputReal, (void*)nullptr);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_OutputDisableInterpolation, (void*)nullptr);

    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MultiFrameCount, in.frames_made);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MultiFrameIndex, in.frame_index);
    // The camera is the whole of what places a made frame in time, and it is dlssg_math.h's: NGX takes row vectors and
    // the depth image's own clip space, neither of which is how the renderer keeps the scene's matrices. Each is a pointer
    // to sixteen floats that must outlive the call, which `in` does.
    const dlssg_math::Camera& c = in.camera;
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip, (void*)c.clip_to_prev_clip);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_PrevClipToClip, (void*)c.prev_clip_to_clip);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraViewToClip, (void*)c.view_to_clip);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_ClipToCameraView, (void*)c.clip_to_view);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_ClipToLensClip, (void*)c.clip_to_lens_clip);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraPosX, c.position[0]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraPosY, c.position[1]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraPosZ, c.position[2]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraUpX, c.up[0]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraUpY, c.up[1]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraUpZ, c.up[2]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraRightX, c.right[0]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraRightY, c.right[1]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraRightZ, c.right[2]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraFwdX, c.forward[0]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraFwdY, c.forward[1]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraFwdZ, c.forward[2]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetX, 0.0f);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetY, 0.0f);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_OrthoProjection, 0u);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MvecDilated, 0u);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MvecJittered, 0u);   // (vk_fsr3_inputs.cpp takes the jitter off them)
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MvecInvalidValue, -FLT_MAX);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MinRelativeLinearDepthObjectSeparation, 40.0f);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_JitterOffsetX, in.jitter[0]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_JitterOffsetY, in.jitter[1]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MvecScaleX, in.motion_scale[0]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_MvecScaleY, in.motion_scale[1]);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraNear, in.near_plane);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraFar, in.far_plane);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraFOV, in.fov_y);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraAspectRatio, in.aspect);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_ColorBuffersHDR, in.colour_hdr ? 1u : 0u);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_DepthInverted, in.depth_inverted ? 1u : 0u);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_CameraMotionIncluded,
                in.camera_motion_included ? 1u : 0u);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_Reset, (in.reset || g.first_evaluation) ? 1u : 0u);
    ngx::set(params, NVSDK_NGX_DLSSG_Parameter_NotRenderingGameFrames, 0u);
    const NVSDK_NGX_Result res = ngx::evaluate_feature(cmd, feature, params);
    if (NVSDK_NGX_FAILED(res)) {
        static int told = 0;
        if (told++ < 4) logf("vk: dlssg: the evaluation was refused (0x%08x)", (unsigned)res);
        return false;
    }
    g.first_evaluation = false;
    return true;
}

void ngx_drop_feature() {
    ngx::release_feature(feature);
    feature = nullptr;
}

void ngx_shutdown() {
    ngx_drop_feature();
    ngx::destroy_params(params);
    params = nullptr;
}

#else   // no NGX SDK: the gate is off, and each of these refuses

bool ngx_available() {
    return false;
}
bool ngx_feature(VkCommandBuffer) {
    return false;
}
bool ngx_generate(VkCommandBuffer, const Inputs&, Image&) {
    return false;
}
void ngx_drop_feature() {}
void ngx_shutdown() {}

#endif

} // namespace

/// What this build and device can do, which cannot change while they run. (A function's static: asked once, whichever
/// thread asks first, the others waiting for the answer.)
bool capable() {
    static const bool yes = []() {
        if (!kHaveSdk) {
            logf("vk: dlssg: this build has no NGX SDK (HOOKSHOT_NGX_SDK), so frame generation is "
                 "unavailable; the renderer's own interpolation is unaffected");
            return false;
        }
        if (!ngx_available()) return false;
        logf("vk: dlssg: frame generation is available on this device, up to %u made frame(s) between two of the game's",
             multi_max.load(std::memory_order_relaxed));
        return true;
    }();
    return yes;
}

bool supported() {
    return env_limit() != 0 && dev.device && capable();
}

uint32_t max_frames() {
    if (!supported()) return 0;
    const uint32_t most = multi_max.load(std::memory_order_relaxed);
    return env_limit() > 0 ? std::min(most, (uint32_t)env_limit()) : most;
}

size_t device_extensions(VkInstance instance, VkPhysicalDevice gpu, const char** out, size_t max) {
#if defined(HOOKSHOT_NGX)
    if (env_limit() == 0) return 0;
    if (const char* e = getenv("RX_VK_DLSS"); e && *e && atoi(e) == 0) return 0;   // (NGX off for this run: vk_dlss.cpp)
    return ngx::device_extensions(instance, gpu, NVSDK_NGX_Feature_FrameGeneration, out, max);
#else
    (void)instance, (void)gpu, (void)out, (void)max;
    return 0;
#endif
}

namespace {

/// The images the made frames resolve into, and the feature's sizes: made again when the sizes, the presented format or
/// the count change. Frames in flight use the old ones, so the device goes idle first (rare: a resize, another multiplier).
bool prepare(uint32_t width, uint32_t height, uint32_t render_width, uint32_t render_height, VkFormat present_format,
             uint32_t n) {
    if (!width || !height || !render_width || !render_height || !n || n > kMaxMade) return false;
    if (present_format == VK_FORMAT_UNDEFINED) return false;
    if (g.ready == n && g.width == width && g.height == height && g.render_width == render_width &&
        g.render_height == render_height && g.present_format == present_format)
        return true;
    // Extents or format changed: the feature was made for the old ones.
    if (g.ready) idle();
    ngx_drop_feature();
    g.width = width, g.height = height;
    g.render_width = render_width, g.render_height = render_height;
    g.present_format = present_format;
    g.first_evaluation = true;
    for (uint32_t i = 0; i < n; i++) {
        if (!make_frame(g.made[i], width, height, present_format)) {
            for (uint32_t k = 0; k <= i; k++) drop_image(g.made[k]);
            g.ready = 0;
            return false;
        }
    }
    for (uint32_t i = n; i < kMaxMade; i++) drop_image(g.made[i]);
    g.ready = n;
    logf("vk: dlssg: %u frame image(s) of %ux%u (format %d), from %ux%u inputs", n, width, height,
         (int)present_format, render_width, render_height);
    return true;
}

}  // namespace

bool generate(VkCommandBuffer cmd, const Inputs& in) {
    if (!supported()) return false;
    if (!in.color || !in.depth || !in.motion) return false;
    if (!in.color_image || !in.depth_image || !in.motion_image) return false;
    if (in.color_format == VK_FORMAT_UNDEFINED || in.depth_format == VK_FORMAT_UNDEFINED ||
        in.motion_format == VK_FORMAT_UNDEFINED)
        return false;
    if (!in.width || !in.height || !in.render_width || !in.render_height) return false;
    if (!in.have_camera) return false;
    if (!in.frames_made || in.frames_made > max_frames() || !in.frame_index || in.frame_index > in.frames_made) return false;
    if (!(in.near_plane > 0.0f) || !(in.fov_y > 0.0f)) return false;   // (Inputs defaults them)
    if (!prepare(in.width, in.height, in.render_width, in.render_height, in.color_format, in.frames_made)) return false;
    if (!ngx_feature(cmd)) return false;
    Image& out = g.made[in.frame_index - 1];
    if (!out.image) return false;
    to_general(cmd, out);
    return ngx_generate(cmd, in, out);
}

VkImage made_image(uint32_t index) {
    if (!index || index > g.ready) return VK_NULL_HANDLE;
    return g.made[index - 1].image;
}

VkImageView made_view(uint32_t index) {
    if (!index || index > g.ready) return VK_NULL_HANDLE;
    return g.made[index - 1].view;
}

void reset() {
    g.first_evaluation = true;
}

void shutdown() {
    ngx_shutdown();
    for (Image& t : g.made) drop_image(t);
    g = State{};
}

} // namespace mkwvk::dlssg
