/* (hookshot fork) The Insert menu's Frame generation method row (opts.framegen_method, vk_menu.cpp) and which method
   gpu/vk/vk_framegen.cpp makes a set of frames with, as plain functions: no Vulkan, no menu.
   tests/test_framegen_methods.cpp checks them.

   A method chosen outright makes the sets it can, where it runs here; a set it cannot make, and a frame it refuses (no
   history yet, a cut, its inputs missing), go to the renderer's own blend, which always runs: a frame is never left out.

   Auto takes NVIDIA DLSS frame generation where it runs and the built-in blend otherwise. It does not take AMD's two.
   None of the vendor methods is given a picture without the HUD (the game draws its HUD into the same picture, and the
   renderer has no copy from before it), and AMD's interpolation then moves the HUD with the scene: upstream measured it
   with FSR 3.1's (a smeared HUD, a trail behind Link: its #1635 and seven duplicates) and made that one opt-in. FSR 4's
   frame generation takes the same inputs through the same kind of pass, so it is a choice here too, not a default.
   NVIDIA's finds the interface in the picture by itself and stays Auto's first.

   How many frames a method makes between two of the game's is the method's own limit on this PC (Here): FSR 3.1's
   interpolation has no phase input (ffx_frameinterpolation.h fixes t at 0.5), so it makes the midpoint and nothing else,
   2x sets only; DLSS makes one, or up to three with multi-frame generation (NGX says which: vk_dlssg.h max_frames); FSR 4
   one unless RX_VK_FFX_FG_MAX says more (vk_ffx.h). A set of more frames than a method makes is not half-made by it -- that
   would show a lower multiplier than the row says, or the same picture twice -- it is the built-in blend's. */
#pragma once

#include <initializer_list>

namespace mkwvk::framegen_methods {

enum { AUTO = 0, DLSS = 1, FSR4 = 2, FSR3 = 3, BUILTIN = 4, kMethods = 5 };

inline unsigned bit(int method) { return 1u << method; }
/// What can always be chosen: Auto and the built-in blend
inline unsigned always() { return bit(AUTO) | bit(BUILTIN); }

/// What runs on this PC: the most frames each vendor method makes between two of the game's, 0 where it does not run
/// (most[AUTO] and most[BUILTIN] are not read: those two make any set)
struct Here {
    int most[kMethods] = {0, 0, 0, 0, 0};
};
/// One bit per method that can be chosen here (bit() of each): what the row steps through
inline unsigned available(const Here& here) {
    unsigned mask = always();
    for (int method : {DLSS, FSR4, FSR3})
        if (here.most[method] > 0) mask |= bit(method);
    return mask;
}

/// The row's next value from v (dir +1 / -1), wrapping over what runs here (available: bit() of each)
inline int step(int v, int dir, unsigned available) {
    available |= always();
    int at = v < 0 || v >= kMethods || !(available & bit(v)) ? AUTO : v;
    for (int i = 0; i < kMethods; i++) {
        at = (at + (dir > 0 ? 1 : kMethods - 1)) % kMethods;
        if (available & bit(at)) return at;
    }
    return AUTO;
}

/// The method shown and used: the one chosen when it runs here, else Auto (a saved choice is kept for a PC where it runs)
inline int effective(int chosen, unsigned available) {
    available |= always();
    return chosen >= 0 && chosen < kMethods && (available & bit(chosen)) ? chosen : AUTO;
}

/// Whether Auto takes a vendor method: NVIDIA DLSS only (the note at the top says why not AMD's)
inline bool in_auto(int method) { return method == DLSS; }

/// Whether a vendor method is the one tried, given the method in effect
inline bool tries(int effective_method, int method) {
    return effective_method == method || (effective_method == AUTO && in_auto(method));
}

/// Whether a vendor method makes a whole set of this multiplier here (2 to 8: multiplier - 1 made frames)
inline bool makes_set(const Here& here, int method, int multiplier) {
    return multiplier >= 2 && (method == DLSS || method == FSR4 || method == FSR3) && multiplier - 1 <= here.most[method];
}

/// The method that makes a set of this multiplier: the vendor method in effect where it runs here and makes such a set,
/// else the built-in blend
inline int first(int effective_method, const Here& here, int multiplier) {
    for (int method : {DLSS, FSR4, FSR3})
        if (tries(effective_method, method) && makes_set(here, method, multiplier)) return method;
    return BUILTIN;
}

} // namespace mkwvk::framegen_methods
