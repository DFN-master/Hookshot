"""Rebuild gpu/vk/host_shaders/bc7_encoder_comp_spv.h from bc7_encoder.comp.

The build uses the checked-in header (it does not run glslang). Set GLSLANG to the Vulkan SDK's
glslangValidator (or glslang, as the 16.x Windows release calls it):
  GLSLANG=C:\\VulkanSDK\\Bin\\glslangValidator.exe python -I scripts/make_bc7_shader.py
"""
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "gpu", "vk", "host_shaders")
COMP = os.path.join(HERE, "bc7_encoder.comp")
HEADER = os.path.join(HERE, "bc7_encoder_comp_spv.h")


def main():
    glslang = os.environ.get("GLSLANG", "glslangValidator")
    with tempfile.TemporaryDirectory() as tmp:
        spv = os.path.join(tmp, "bc7_encoder.comp.spv")
        subprocess.check_call([glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-o", spv, COMP])
        data = open(spv, "rb").read()
    if len(data) % 4:
        sys.exit("the SPIR-V is not a whole number of 32-bit words")
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    lines = ["#pragma once", "#include <cstdint>", "", "const uint32_t BC7_ENCODER_COMP_SPV[] = {"]
    row = []
    for word in words:
        row.append("0x%08x," % word)
        if len(row) == 8:
            lines.append("\t" + "".join(row))
            row = []
    if row:
        lines.append("\t" + "".join(row))
    lines.append("};")
    with open(HEADER, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s (%d words, %d bytes)" % (HEADER, len(words), len(data)))


if __name__ == "__main__":
    main()
