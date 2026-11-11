// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Sustituto de common/settings.h de Eden para su shader_recompiler: solo los valores que lee el
// recompilador, con los valores por defecto del emulador (sin escalado de resolución).

#include "common/common_types.h"

namespace Settings {

struct ResolutionScalingInfo {
    u32 up_scale = 1;
    u32 down_shift = 0;
    f32 up_factor = 1.0f;
    f32 down_factor = 1.0f;
    bool active = false;
};

struct Values {
    bool renderer_debug = false;
    bool disable_shader_loop_safety_checks = false;
    ResolutionScalingInfo resolution_info{};
};

inline Values values;

}  // namespace Settings
