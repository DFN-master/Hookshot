"""NVIDIA DLSS frame generation's camera without a game or GPU (gpu/vk/dlssg_math.h, checked by behaviour in
tests/test_dlssg_math.cpp): the matrices as NGX takes them (row vectors, the depth image's clip space), the camera's
vectors, the motion vectors' scale and the jitter; and that vk_framegen.cpp hands NGX those rather than the renderer's own
matrices as they are.

python -I -B tests/test_dlssg_math.py [--cxx g++]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

RUN_ALL_ARGS = ["--cxx", "g++"]

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dlssg-math-") as tmp:
        exe = Path(tmp) / "dlssg-math"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", str(ROOT / "tests" / "test_dlssg_math.cpp"),
                        "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    fg = (ROOT / "gpu" / "vk" / "vk_framegen.cpp").read_text(encoding="utf-8")
    start = fg.index("int dlssg_frames(")
    body = fg[start:fg.index("\n}\n", start)]
    for needle, what in (
            ("dlssg_math::camera(f.projection, f.view_projection, f.previous_view_projection, f.camera.flip_depth",
             "the camera through dlssg_math.h, with the depth copy's direction"),
            ("dlssg_math::motion_scale(f.width, f.height", "motion vectors normalized by the render size"),
            ("dlssg_math::jitter_to_clip(f.jitter, f.width, f.height", "the jitter in clip space")):
        if needle not in body:
            raise SystemExit("vk_framegen.cpp dlssg_frames: missing %s (%s)" % (what, needle))
    if "f.projection, sizeof" in body or "in.motion_scale[0] = 1.0f" in body:
        raise SystemExit("vk_framegen.cpp dlssg_frames hands NGX the renderer's own matrices or pixel motion as they are")
    print("PASS: vk_framegen.cpp gives DLSS frame generation dlssg_math.h's camera")


if __name__ == "__main__":
    main()
