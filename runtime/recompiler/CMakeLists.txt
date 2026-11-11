# eden_shader: el recompilador de shaders de Eden (third_party/eden, GPL-3) como librería estática,
# solo con el backend SPIR-V. De su common/ usa las cabeceras; settings/logging/assert se sustituyen
# por las de shim/, que van antes en el include path.
#
# Lo usan tools/shadertool (en el Mac) y libnnsdk.so (en la Flip). Hace GPL-3 a lo que lo enlaza.

set(TP ${CMAKE_CURRENT_LIST_DIR}/../../third_party)

set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/fmt ${CMAKE_BINARY_DIR}/fmt EXCLUDE_FROM_ALL)

set(SIRIT_TESTS OFF CACHE BOOL "" FORCE)
set(SIRIT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/sirit ${CMAKE_BINARY_DIR}/sirit EXCLUDE_FROM_ALL)

# SPIRV-Tools (Apache-2.0), de las fuentes del NDK: el optimizador por el que pasa el SPIR-V de Eden (spirv_opt.cpp).
include(${CMAKE_CURRENT_LIST_DIR}/spirv_tools.cmake)

file(GLOB_RECURSE EDEN_SHADER_SOURCES ${TP}/eden/src/shader_recompiler/*.cpp)
list(FILTER EDEN_SHADER_SOURCES EXCLUDE REGEX "/backend/(glsl|glasm)/")
add_library(eden_shader STATIC ${EDEN_SHADER_SOURCES})
target_compile_features(eden_shader PUBLIC cxx_std_20)
target_include_directories(eden_shader BEFORE PUBLIC ${CMAKE_CURRENT_LIST_DIR}/shim)
target_include_directories(eden_shader PUBLIC ${TP}/eden/src)
file(GLOB BOOST_INCLUDES ${TP}/boost/libs/*/include)
target_include_directories(eden_shader SYSTEM PUBLIC ${BOOST_INCLUDES})
target_link_libraries(eden_shader PUBLIC fmt::fmt sirit::sirit)
target_compile_options(eden_shader PRIVATE -fbracket-depth=1024 -w)

# Puente NVN -> recompilador (nuestro, GPL-3 por enlazar con Eden): lo que usan libnnsdk y shadertool.
target_sources(eden_shader PRIVATE ${CMAKE_CURRENT_LIST_DIR}/nvn_shader.cpp ${CMAKE_CURRENT_LIST_DIR}/spirv_opt.cpp)
target_include_directories(eden_shader PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(eden_shader PRIVATE spirv_tools_opt)

# Huella del traductor para la caché de SPIR-V de la SD (<datos>/spirv.bin, ver nnsdk/nvn/spirv_cache.h): el contenido
# de todo lo que decide una traducción. Eden ya parcheado (tools/patches se aplica sobre third_party), el puente y su
# shim, sirit y SPIRV-Tools. Si cambia cualquiera, CMake se vuelve a configurar (CMAKE_CONFIGURE_DEPENDS; apk.sh configura siempre)
# y con la huella nueva el fichero se vacía al arrancar. El .h solo se reescribe si cambia la huella.
file(GLOB_RECURSE NVN_TRADUCTOR_FICHEROS
  ${TP}/eden/src/shader_recompiler/*.cpp ${TP}/eden/src/shader_recompiler/*.h ${TP}/eden/src/shader_recompiler/*.inc
  ${TP}/sirit/src/*.cpp ${TP}/sirit/src/*.h ${TP}/sirit/include/*.h
  ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.cmake
  ${CMAKE_CURRENT_LIST_DIR}/shim/*)
list(SORT NVN_TRADUCTOR_FICHEROS)
set(NVN_TRADUCTOR_TODO "")
foreach(f IN LISTS NVN_TRADUCTOR_FICHEROS)
  file(SHA256 ${f} h)
  file(RELATIVE_PATH rel ${TP}/.. ${f})
  string(APPEND NVN_TRADUCTOR_TODO "${rel} ${h}\n")
endforeach()
# SPIRV-Tools también decide lo que sale (Options::optimize_stages): sus fuentes y las tablas generadas de su gramática,
# con rutas relativas a ellas (la del NDK es de cada máquina).
file(GLOB NVN_TRADUCTOR_SPVT ${SPIRV_TOOLS_DIR}/CHANGES ${SPIRV_TOOLS_DIR}/source/*.cpp ${SPIRV_TOOLS_DIR}/source/*.h
  ${SPIRV_TOOLS_DIR}/source/opt/* ${SPIRV_TOOLS_DIR}/source/util/* ${SPIRV_TOOLS_DIR}/source/val/*
  ${SPIRV_TOOLS_DIR}/include/spirv-tools/*)
file(GLOB NVN_TRADUCTOR_SPVT_GEN ${SPIRV_TOOLS_GEN}/*.inc ${SPIRV_TOOLS_GEN}/*.h)
list(SORT NVN_TRADUCTOR_SPVT)
list(SORT NVN_TRADUCTOR_SPVT_GEN)
foreach(f IN LISTS NVN_TRADUCTOR_SPVT NVN_TRADUCTOR_SPVT_GEN)
  file(SHA256 ${f} h)
  string(FIND "${f}" "${SPIRV_TOOLS_GEN}/" at)
  if(at EQUAL 0)
    file(RELATIVE_PATH rel ${SPIRV_TOOLS_GEN} ${f})
    string(PREPEND rel "spirv-tools-gen/")
  else()
    file(RELATIVE_PATH rel ${SPIRV_TOOLS_DIR} ${f})
    string(PREPEND rel "spirv-tools/")
  endif()
  string(APPEND NVN_TRADUCTOR_TODO "${rel} ${h}\n")
  list(APPEND NVN_TRADUCTOR_FICHEROS ${f})
endforeach()
string(SHA256 NVN_TRADUCTOR_TODO "${NVN_TRADUCTOR_TODO}")
string(SUBSTRING ${NVN_TRADUCTOR_TODO} 0 16 NVN_TRADUCTOR_HUELLA)
list(LENGTH NVN_TRADUCTOR_FICHEROS NVN_TRADUCTOR_N)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${NVN_TRADUCTOR_FICHEROS})
file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/generado/nvn_traductor_huella.h CONTENT
  "#pragma once\n// Generado por runtime/recompiler/CMakeLists.txt (${NVN_TRADUCTOR_N} ficheros del traductor).\n#define NVN_TRADUCTOR_HUELLA 0x${NVN_TRADUCTOR_HUELLA}ULL\n")
target_include_directories(eden_shader PUBLIC ${CMAKE_BINARY_DIR}/generado)
message(STATUS "Huella del traductor de shaders: ${NVN_TRADUCTOR_HUELLA} (${NVN_TRADUCTOR_N} ficheros)")
