// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 lsfg-vk
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once
// Lossless.dll del jugador (Lossless Scaling): se lee como bytes (nunca se carga) y de su .rsrc salen los
// shaders SPIR-V de la generación de frames. Port de Eden (src/video_core/frame_gen/lossless_dll.{h,cpp} y
// lsfg_translate.cpp). La DLL y todo lo que sale de ella se quedan en el dispositivo: nada va al repo ni a la APK.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nvn::lsfg {

// Ids de los shaders del modo rendimiento (la tabla de Eden). Cada id tiene su SPIR-V FP16 en id + 49 y el
// FP32 en id + 98.
namespace shader {
constexpr uint32_t kMipmaps = 255;
constexpr uint32_t kGenerate = 256;
constexpr uint32_t kAlpha[4] = {290, 291, 292, 293};
constexpr uint32_t kBeta[5] = {298, 299, 300, 301, 302};
constexpr uint32_t kGamma[5] = {280, 282, 283, 284, 285};
constexpr uint32_t kDelta[10] = {280, 286, 287, 288, 289, 281, 294, 295, 296, 297};
constexpr uint32_t kOffsetFp16 = 49, kOffsetFp32 = 98;
}  // namespace shader

enum class DllStatus { Ok, NotFound, Unreadable, NotPortableExecutable, MissingShaders };
const char* dll_status_name(DllStatus s);

// Dónde está la DLL: <NIER_DATA>/lsfg/Lossless.dll y si no /storage/emulated/0/Download/Lossless.dll; "" si
// no hay ninguna legible.
std::string find_dll();

// id de rendimiento (255, 256, 280..302) -> SPIR-V con los bindings renumerados 0..N-1 en orden (set, binding),
// como los espera la tabla de descriptores (UBO, samplers, imágenes muestreadas, imágenes de storage).
using Modules = std::map<uint32_t, std::vector<uint32_t>>;

// Lee la DLL y saca los dos juegos (FP16 y FP32). Un juego al que le falta algún shader queda vacío.
DllStatus load_shaders(const std::string& path, Modules& fp16, Modules& fp32, size_t* dll_bytes);

// Comprueba que el módulo vale en nuestro dispositivo (SPIR-V <= 1.3, modelo de memoria GLSL450, solo
// capacidades que tenemos; Float16 solo si `float16`) y saca su punto de entrada y los tipos de sus bindings
// (VK_DESCRIPTOR_TYPE_*, por número de binding). false con el motivo en `why`.
bool audit_module(const std::vector<uint32_t>& words, bool float16, std::string* entry, std::vector<int>* bindings,
                  std::string* why);

}  // namespace nvn::lsfg
