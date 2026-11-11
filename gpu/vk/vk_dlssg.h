// (hookshot) NVIDIA DLSS Frame Generation as a source of the frames shown between the game's.
//
// The renderer already shows made frames: vk_framegen.cpp interpolates between the last two
// presented frames and the display thread paces them (gpu/vk/vk_framegen.h). That interpolation is
// made from colour alone, which is all it has. This backend answers the same question with the
// game's motion and depth and NVIDIA's network, and with MultiFrameCount it can make more than one
// frame between two real ones -- which is the only way past a frame rate the game's own simulation
// limits.
//
// It slots in where the hand-written interpolation does, and where it refuses -- no NGX, no runtime,
// a GPU that does not report Frame Generation, inputs the seam could not assemble -- that
// interpolation runs exactly as before. Nothing here changes what the game submits.
//
// Nothing of NVIDIA's is in this repository and nothing of NVIDIA's is needed to build it: the NGX
// half is inside `HOOKSHOT_NGX` and the session is shared (gpu/vk/vk_ngx.h). docs/upscaling.md has
// the licensing decision and the input contract.
#pragma once

#include <cstdint>

#include "dlssg_math.h"
#include "vk_internal.h"

namespace mkwvk::dlssg {

/// One evaluation's inputs.
///
/// The colour is the frame the game has just finished, at the size it is presented; the motion and depth are the scene's,
/// at the render size (vk_fsr3_inputs.h Frame: the depth reversed, the motion in render pixels toward the frame before,
/// without the jitter). The camera is dlssg_math.h's: the scene's matrices as NGX takes them, which is not as the renderer
/// keeps them (row vectors, the depth image's clip space); `motion_scale` and `jitter` are dlssg_math.h's too.
struct Inputs {
    VkImageView color = VK_NULL_HANDLE;
    VkImage color_image = VK_NULL_HANDLE;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;

    VkImageView depth = VK_NULL_HANDLE;
    VkImage depth_image = VK_NULL_HANDLE;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkImageView motion = VK_NULL_HANDLE;
    VkImage motion_image = VK_NULL_HANDLE;
    VkFormat motion_format = VK_FORMAT_UNDEFINED;
    uint32_t render_width = 0, render_height = 0;

    float motion_scale[2]{};   ///< normalizes a motion vector (dlssg_math.h motion_scale)
    float jitter[2]{};         ///< in clip space (dlssg_math.h jitter_to_clip)
    dlssg_math::Camera camera;
    /// An evaluation without the whole camera is refused by NGX (NVSDK_NGX_Result_FAIL_MissingInput, which a run saw), so
    /// one without it is not made here either: generate() answers false
    bool have_camera = false;

    float near_plane = 0.1f;
    float far_plane = 1000.0f;
    float fov_y = 60.0f * 3.14159265358979f / 180.0f;
    float aspect = 16.0f / 9.0f;
    bool depth_inverted = true;
    bool colour_hdr = false;
    /// The motion already includes the camera's movement. Ours does: it *is* the camera's movement
    /// (gpu/vk/vk_fsr3_inputs.h, shaders/fsr3_prep.comp), which is why this defaults true and why
    /// saying otherwise would have the network apply the camera twice.
    bool camera_motion_included = true;
    bool reset = false;

    /// How many frames are made between this frame and the last (1 to max_frames()), and which of them this evaluation is
    /// (1-based). 1 and 1 is ordinary 2x generation; more is DLSS 4's multi-frame generation, an evaluation a made frame.
    uint32_t frames_made = 1;
    uint32_t frame_index = 1;
};

/// Whether frame generation can run here at all: built with an NGX SDK, the NGX session up, this driver and GPU reporting
/// Frame Generation, and RX_VK_DLSSG not 0. NGX is asked once (and may take seconds to start: vk_menu.cpp asks during the
/// start-up screen, so the Insert menu and the first frame only read the answer); the one line saying why not is logged
/// once. It does not depend on the Frame generation row: the method row offers what runs.
[[nodiscard]] bool supported();

/// The most frames it makes between two of the game's: NGX's own limit for this GPU (DLSSG.MultiFrameCountMax: 1 without
/// multi-frame generation, 3 with it), under RX_VK_DLSSG=<1-3> when that is set. 0 where it is not supported(). A set of
/// more made frames than this is not its to make (framegen_methods.h makes_set).
[[nodiscard]] uint32_t max_frames();

/// The device extensions frame generation needs here, asked of NGX itself before the device exists.
/// Super resolution's set is not the same set: a device made for one and not the other is where
/// `NVSDK_NGX_VULKAN_CreateFeature` answers FAIL_PlatformError with the feature still reporting
/// itself available, which is exactly what a run saw. Asked whatever the Frame generation row says: the device is made
/// before a player can turn the row on, and a device without these could never run it this session. Returns how many
/// names were written, at most `max`; zero with RX_VK_DLSSG=0 or RX_VK_DLSS=0 (which leaves NGX off), in a build without
/// an NGX SDK, or where NGX does not answer. The names are NGX's own storage and last as long as the process.
size_t device_extensions(VkInstance instance, VkPhysicalDevice gpu, const char** out, size_t max);

/// Records one made frame, making the feature and the frames' images for these sizes first (again when the sizes, the
/// presented format or the count change, with the device idle). False when the inputs are not usable or NGX refused,
/// which leaves the caller to show whatever it would otherwise have shown.
[[nodiscard]] bool generate(VkCommandBuffer cmd, const Inputs& in);

/// The made frame of the last `generate`, by index (1-based, as `Inputs::frame_index`), or null.
[[nodiscard]] VkImage made_image(uint32_t index);
[[nodiscard]] VkImageView made_view(uint32_t index);

/// Forgets the sequence, so the next evaluation is told to reset: a cut, a load, a resize.
void reset();

/// Frees everything this backend made. The shared NGX session is not ended here.
void shutdown();

} // namespace mkwvk::dlssg
