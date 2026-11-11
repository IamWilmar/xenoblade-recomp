# Third-party notices

This project is licensed under GPL-3.0-or-later. It uses or includes the following third-party components.

| Component | Use | License |
|---|---|---|
| [Eden](https://git.eden-emu.dev/eden-emu/eden) (yuzu fork) | Shader recompiler (`src/shader_recompiler`, `src/common`), fetched by `tools/setup.sh` and patched with `tools/patches/` | GPL-3.0-or-later |
| [lsfg-vk](https://github.com/PancakeTAS/lsfg-vk) (via Eden) | Frame-generation integration in `runtime/src/nnsdk/nvn/lsfg/` (loads the user's own `Lossless.dll`; no Lossless Scaling code or shaders are included) | GPL-3.0-or-later |
| [AMD FidelityFX Super Resolution 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR) | Upscaling shaders (`runtime/src/nnsdk/nvn/shaders/`, compiled into `spirv.h`) | MIT, Copyright (c) 2021 Advanced Micro Devices, Inc. |
| [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) | SPIR-V optimizer, built from the copy shipped in the Android NDK | Apache-2.0 (see `runtime/recompiler/LICENSE.SPIRV-Tools`) |
| [sirit](https://github.com/eden-emulator/sirit) | SPIR-V emitter used by the shader recompiler | BSD-3-Clause |
| [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers) | SPIR-V grammar headers | MIT-style (Khronos) |
| [{fmt}](https://github.com/fmtlib/fmt) | Formatting library used by Eden's code | MIT |
| [Boost](https://www.boost.org/) (headers) | Used by Eden's shader recompiler | BSL-1.0 |
| [libopus](https://opus-codec.org/) | Opus audio decoding (nn::codec) | BSD-3-Clause |
| Android NDK `native_app_glue` | NativeActivity glue | Apache-2.0 |

Lossless Scaling and its `Lossless.dll` are not distributed here; the DLL must come from the user's own purchase.

AMD FidelityFX FSR 1 license (MIT):

```
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
