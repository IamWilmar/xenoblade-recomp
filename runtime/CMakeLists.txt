cmake_minimum_required(VERSION 3.22)
project(nier_runtime C CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

# libnnsdk.so: nuestra reimplementación del SDK de Nintendo. Todo símbolo que
# define se enlaza al juego con prioridad sobre bionic y sobre los stubs.
add_library(nnsdk SHARED
  src/nnsdk/account.cpp
  src/nnsdk/audio.cpp
  src/nnsdk/codec.cpp
  src/nnsdk/cxx.cpp
  src/nnsdk/diag.cpp
  src/nnsdk/fs.cpp
  src/nnsdk/hid.cpp
  src/nnsdk/libc.cpp
  src/nnsdk/mem.cpp
  src/nnsdk/movie.cpp
  src/nnsdk/oe.cpp
  src/nnsdk/swkbd.cpp
  src/nnsdk/time.cpp
  src/nnsdk/nvn/buffer.cpp
  src/nnsdk/nvn/device.cpp
  src/nnsdk/nvn/cmdbuf.cpp
  src/nnsdk/nvn/formats.cpp
  src/nnsdk/nvn/executor.cpp
  src/nnsdk/nvn/gpu_memory.cpp
  src/nnsdk/nvn/memory.cpp
  src/nnsdk/nvn/layout.cpp
  src/nnsdk/nvn/lsfg/chain.cpp
  src/nnsdk/nvn/lsfg/dll.cpp
  src/nnsdk/nvn/lsfg/framegen.cpp
  src/nnsdk/nvn/state.cpp
  src/nnsdk/nvn/procs.cpp
  src/nnsdk/nvn/program.cpp
  src/nnsdk/nvn/spirv_cache.cpp
  src/nnsdk/nvn/queue.cpp
  src/nnsdk/nvn/texture.cpp
  src/nnsdk/nvn/vk.cpp
  src/nnsdk/nvn/window.cpp
  src/nnsdk/nvn/hud.cpp
  src/nnsdk/nvn/fsr.cpp
  src/nnsdk/os.cpp
  src/nnsdk/vi.cpp
  src/nnsdk/aoc.cpp
  src/nnsdk/album.cpp
  src/nnsdk/bcat.cpp
  src/nnsdk/crypto.cpp
  src/nnsdk/cxx_ios.cpp
  src/nnsdk/ec.cpp
  src/nnsdk/err.cpp
  src/nnsdk/friends.cpp
  src/nnsdk/hid_vibration.cpp
  src/nnsdk/image.cpp
  src/nnsdk/ldn.cpp
  src/nnsdk/libc_file.cpp
  src/nnsdk/libc_thread.cpp
  src/nnsdk/nfp.cpp
  src/nnsdk/ngc.cpp
  src/nnsdk/nifm.cpp
  src/nnsdk/nsd.cpp
  src/nnsdk/os_fiber.cpp
  src/nnsdk/os_memory.cpp
  src/nnsdk/watch.cpp
  src/nnsdk/os_sync.cpp
  src/nnsdk/pctl.cpp
  src/nnsdk/prepo.cpp
  src/nnsdk/settings.cpp
  src/nnsdk/socket.cpp
  src/nnsdk/ssl.cpp
  src/nnsdk/util.cpp
  src/nnsdk/web.cpp
)
target_include_directories(nnsdk PRIVATE src)
# Recompilador de shaders de Eden (GPL-3): libnnsdk.so queda bajo GPL-3.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(recompiler)
target_link_libraries(nnsdk PRIVATE eden_shader)
# libopus (BSD-3, third_party/opus con tools/setup.sh): el HardwareOpusDecoder de nn::codec (codec.cpp), con el que
# Wwise decodifica todo el Opus NX del juego (voces, música). Estática y sin exportar (--exclude-libs,ALL).
set(OPUS_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/opus)
if(NOT EXISTS ${OPUS_DIR}/CMakeLists.txt)
  message(FATAL_ERROR "Falta third_party/opus (libopus v1.6.1): tools/setup.sh")
endif()
foreach(opcion OPUS_BUILD_SHARED_LIBRARY OPUS_BUILD_TESTING OPUS_BUILD_PROGRAMS OPUS_INSTALL_PKG_CONFIG_MODULE
               OPUS_INSTALL_CMAKE_CONFIG_MODULE OPUS_CUSTOM_MODES OPUS_DRED OPUS_OSCE OPUS_DEEP_PLC)
  set(${opcion} OFF CACHE BOOL "" FORCE)
endforeach()
set(OPUS_PRESUME_NEON ON CACHE BOOL "" FORCE)  # arm64: NEON siempre, sin detección en tiempo de ejecución
add_subdirectory(${OPUS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/opus EXCLUDE_FROM_ALL)
target_link_libraries(nnsdk PRIVATE opus)
target_link_libraries(nnsdk PRIVATE vulkan android mediandk aaudio z)
# Su libc++ estática no se exporta: el juego debe usar una sola copia de libc++abi (la del loader).
target_link_options(nnsdk PRIVATE -Wl,--exclude-libs,ALL)

# Núcleo del loader: carga main, lo reloca contra libnnsdk/bionic/stubs y lo arranca.
add_library(loader_core OBJECT
  src/loader/run.cpp
  src/loader/nso.cpp
  src/loader/imports.cpp
  src/loader/crash.cpp
  src/loader/host_symbols.c
  src/loader/stub_dispatch.S
  src/loader/tls.cpp
  src/loader/tls_slow.S
  src/loader/probe.cpp
  src/loader/probe_slow.S
)
target_include_directories(loader_core PUBLIC src)
set_target_properties(loader_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# nier_loader: desde la terminal de la Flip, sin ventana.
add_executable(nier_loader src/loader/main.cpp)
target_link_libraries(nier_loader PRIVATE loader_core dl)

# libnier.so: la APK (NativeActivity), con ventana.
set(GLUE ${ANDROID_NDK}/sources/android/native_app_glue)
add_library(nier SHARED src/loader/android_main.cpp ${GLUE}/android_native_app_glue.c)
target_include_directories(nier PRIVATE ${GLUE})
target_link_libraries(nier PRIVATE loader_core android log dl)
target_link_options(nier PRIVATE -u ANativeActivity_onCreate)
