// (hookshot) gpu/vk/framegen_methods.h: the Frame generation method row's stepping and what vk_framegen.cpp tries.
#include <cstdio>
#include "../gpu/vk/framegen_methods.h"

namespace fgm = mkwvk::framegen_methods;
static int failures;
#define CHECK(name, c) do { if (!(c)) { std::printf("FAIL: %s\n", name); ++failures; } } while (0)

int main() {
    using namespace fgm;
    const unsigned all = bit(AUTO) | bit(DLSS) | bit(FSR4) | bit(FSR3) | bit(BUILTIN);
    const unsigned fsr3 = bit(FSR3);   // (Auto and Built-in are always there)
    CHECK("steps through all five", step(AUTO, 1, all) == DLSS && step(DLSS, 1, all) == FSR4 && step(FSR4, 1, all) == FSR3 &&
                                        step(FSR3, 1, all) == BUILTIN && step(BUILTIN, 1, all) == AUTO);
    CHECK("steps back", step(AUTO, -1, all) == BUILTIN && step(DLSS, -1, all) == AUTO);
    CHECK("skips what does not run", step(AUTO, 1, fsr3) == FSR3 && step(FSR3, 1, fsr3) == BUILTIN &&
                                         step(BUILTIN, 1, fsr3) == AUTO && step(AUTO, -1, fsr3) == BUILTIN);
    CHECK("nothing but the always ones", step(AUTO, 1, 0) == BUILTIN && step(BUILTIN, 1, 0) == AUTO);
    CHECK("a bad value steps from Auto", step(9, 1, all) == DLSS && step(-3, -1, all) == BUILTIN);
    CHECK("a choice that runs is kept", effective(FSR3, fsr3) == FSR3 && effective(BUILTIN, 0) == BUILTIN);
    CHECK("a choice that does not run is Auto", effective(DLSS, fsr3) == AUTO && effective(FSR4, 0) == AUTO &&
                                                    effective(7, all) == AUTO);
    // Auto takes NVIDIA's and neither of AMD's: those move the HUD with the scene (framegen_methods.h says how that is known)
    CHECK("Auto takes DLSS only", tries(AUTO, DLSS) && !tries(AUTO, FSR4) && !tries(AUTO, FSR3) && in_auto(DLSS) &&
                                      !in_auto(FSR4) && !in_auto(FSR3) && !in_auto(BUILTIN));
    CHECK("a chosen method alone", tries(FSR3, FSR3) && !tries(FSR3, DLSS) && !tries(FSR3, FSR4) && tries(FSR4, FSR4) &&
                                       !tries(FSR4, DLSS) && tries(DLSS, DLSS));
    CHECK("built-in tries no vendor method", !tries(BUILTIN, DLSS) && !tries(BUILTIN, FSR4) && !tries(BUILTIN, FSR3));
    // what runs here: the most frames each makes between two of the game's (0: it does not run)
    Here none, rtx40, rtx50, amd, every;
    rtx40.most[DLSS] = 1, rtx40.most[FSR3] = 1;
    rtx50.most[DLSS] = 3, rtx50.most[FSR3] = 1;
    amd.most[FSR4] = 1, amd.most[FSR3] = 1;
    every.most[DLSS] = 3, every.most[FSR4] = 4, every.most[FSR3] = 1;
    CHECK("the row offers what runs", available(none) == always() && available(rtx40) == (always() | bit(DLSS) | bit(FSR3)) &&
                                          available(amd) == (always() | bit(FSR4) | bit(FSR3)) && available(every) == all);
    // a method makes a set only whole: multiplier - 1 frames, within what it makes here
    CHECK("FSR 3.1 makes 2x sets only", makes_set(every, FSR3, 2) && !makes_set(every, FSR3, 3) && !makes_set(every, FSR3, 8));
    CHECK("DLSS without multi-frame generation makes 2x only", makes_set(rtx40, DLSS, 2) && !makes_set(rtx40, DLSS, 3));
    CHECK("DLSS with it makes up to 4x", makes_set(rtx50, DLSS, 2) && makes_set(rtx50, DLSS, 4) && !makes_set(rtx50, DLSS, 5) &&
                                             !makes_set(rtx50, DLSS, 8));
    CHECK("FSR 4 within its own limit", makes_set(amd, FSR4, 2) && !makes_set(amd, FSR4, 3) && makes_set(every, FSR4, 5) &&
                                            !makes_set(every, FSR4, 6));
    CHECK("what does not run makes nothing", !makes_set(none, DLSS, 2) && !makes_set(amd, DLSS, 2) && !makes_set(rtx40, FSR4, 2));
    CHECK("no set under 2x, and none by Auto or the built-in blend themselves",
          !makes_set(every, DLSS, 1) && !makes_set(every, DLSS, 0) && !makes_set(every, AUTO, 2) && !makes_set(every, BUILTIN, 2));
    // which method makes a set
    CHECK("Auto: DLSS where it runs and makes the set", first(AUTO, rtx50, 2) == DLSS && first(AUTO, rtx50, 4) == DLSS &&
                                                            first(AUTO, rtx40, 2) == DLSS);
    CHECK("Auto: the built-in blend past what DLSS makes", first(AUTO, rtx50, 5) == BUILTIN && first(AUTO, rtx40, 3) == BUILTIN);
    CHECK("Auto: never AMD's", first(AUTO, amd, 2) == BUILTIN && first(AUTO, none, 2) == BUILTIN);
    CHECK("a chosen AMD method makes its sets", first(FSR4, amd, 2) == FSR4 && first(FSR3, amd, 2) == FSR3 &&
                                                    first(FSR4, every, 5) == FSR4);
    CHECK("and leaves the others to the built-in blend", first(FSR4, amd, 3) == BUILTIN && first(FSR3, every, 4) == BUILTIN &&
                                                             first(DLSS, rtx40, 3) == BUILTIN);
    CHECK("built-in and what does not run", first(BUILTIN, every, 2) == BUILTIN && first(DLSS, amd, 2) == BUILTIN &&
                                                first(FSR4, rtx50, 2) == BUILTIN);
    if (failures) return 1;
    std::printf("PASS: the Frame generation method row steps over what runs, and a method makes only the sets it makes whole\n");
    return 0;
}
