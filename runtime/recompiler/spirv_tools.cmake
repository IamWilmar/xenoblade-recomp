# spirv_tools_opt: SPIRV-Tools (Khronos/Google, Apache-2.0: ver LICENSE.SPIRV-Tools) como librería estática, con el
# núcleo (binario, tablas de la gramática, validador) y el optimizador de spirv-opt. Lo usa spirv_opt.cpp para pasar por
# él el SPIR-V que sale de Eden (ver nvnshader::Options::optimize_stages).
#
# Se compila de las fuentes que trae el NDK (sources/third_party/shaderc/third_party/spirv-tools: v2022.4, las mismas del
# spirv-opt de ndk/shader-tools), sin descargar nada. Las tablas de la gramática
# (.inc y cabeceras de los conjuntos extendidos) las generan sus scripts de utils/ al configurar, como hace su Android.mk,
# con los JSON de su external/spirv-headers (third_party/SPIRV-Headers, el de sirit, es otra versión: aquí no se usa).
# Se generan una vez por carpeta de fuentes (un .fuentes en el directorio de salida): reconfigurar no recompila nada.
#
# Define SPIRV_TOOLS_DIR (las fuentes), SPIRV_TOOLS_VERSION y SPIRV_TOOLS_GEN (lo generado).

if(TARGET spirv_tools_opt)
  return()
endif()

if(ANDROID_NDK)
  set(_spvt_ndk ${ANDROID_NDK})
elseif(DEFINED ENV{ANDROID_NDK_HOME})
  set(_spvt_ndk $ENV{ANDROID_NDK_HOME})
elseif(DEFINED ENV{ANDROID_HOME})
  set(_spvt_ndk $ENV{ANDROID_HOME}/ndk/28.2.13676358)
else()
  set(_spvt_ndk $ENV{HOME}/Library/Android/sdk/ndk/28.2.13676358)  # el de tools/apk.sh
endif()
set(SPIRV_TOOLS_DIR ${_spvt_ndk}/sources/third_party/shaderc/third_party/spirv-tools CACHE PATH
    "Fuentes de SPIRV-Tools (las del NDK)")
if(NOT EXISTS ${SPIRV_TOOLS_DIR}/source/opt/optimizer.cpp)
  message(FATAL_ERROR "SPIRV-Tools: no están sus fuentes en ${SPIRV_TOOLS_DIR} (-DSPIRV_TOOLS_DIR=...)")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_spvt ${SPIRV_TOOLS_DIR})
set(_spvt_headers ${_spvt}/external/spirv-headers/include)
set(_spvt_grammar ${_spvt_headers}/spirv/unified1)
set(SPIRV_TOOLS_GEN ${CMAKE_BINARY_DIR}/spirv-tools-gen)

# La versión, de su CHANGES (la primera línea "vAAAA.N fecha"), como su update_build_version.py pero sin git.
file(STRINGS ${_spvt}/CHANGES _spvt_line REGEX "^v[0-9]+\\.[0-9]+(-dev)? [0-9][0-9][0-9][0-9]-" LIMIT_COUNT 1)
string(REGEX REPLACE " .*" "" SPIRV_TOOLS_VERSION "${_spvt_line}")

set(_spvt_stamp ${SPIRV_TOOLS_GEN}/.fuentes)
set(_spvt_have "")
if(EXISTS ${_spvt_stamp})
  file(READ ${_spvt_stamp} _spvt_have)
endif()
if(NOT _spvt_have STREQUAL _spvt)
  file(REMOVE_RECURSE ${SPIRV_TOOLS_GEN})
  file(MAKE_DIRECTORY ${SPIRV_TOOLS_GEN})
  macro(_spvt_run)
    execute_process(COMMAND ${Python3_EXECUTABLE} ${ARGN} RESULT_VARIABLE _spvt_r OUTPUT_QUIET)
    if(_spvt_r)
      message(FATAL_ERROR "SPIRV-Tools: falló ${ARGN}")
    endif()
  endmacro()
  set(_g ${SPIRV_TOOLS_GEN})
  _spvt_run(${_spvt}/utils/generate_grammar_tables.py
    --spirv-core-grammar=${_spvt_grammar}/spirv.core.grammar.json
    --extinst-glsl-grammar=${_spvt_grammar}/extinst.glsl.std.450.grammar.json
    --extinst-opencl-grammar=${_spvt_grammar}/extinst.opencl.std.100.grammar.json
    --extinst-debuginfo-grammar=${_spvt_grammar}/extinst.debuginfo.grammar.json
    --extinst-cldebuginfo100-grammar=${_spvt_grammar}/extinst.opencl.debuginfo.100.grammar.json
    --core-insts-output=${_g}/core.insts-unified1.inc
    --glsl-insts-output=${_g}/glsl.std.450.insts.inc
    --opencl-insts-output=${_g}/opencl.std.insts.inc
    --operand-kinds-output=${_g}/operand.kinds-unified1.inc)
  _spvt_run(${_spvt}/utils/generate_grammar_tables.py
    --spirv-core-grammar=${_spvt_grammar}/spirv.core.grammar.json
    --extinst-debuginfo-grammar=${_spvt_grammar}/extinst.debuginfo.grammar.json
    --extinst-cldebuginfo100-grammar=${_spvt_grammar}/extinst.opencl.debuginfo.100.grammar.json
    --extension-enum-output=${_g}/extension_enum.inc
    --enum-string-mapping-output=${_g}/enum_string_mapping.inc)
  # Conjuntos extendidos de vendedores y de depuración: <nombre>=<prefijo de sus tipos de operando>.
  foreach(_v debuginfo= opencl.debuginfo.100=CLDEBUG100_ nonsemantic.shader.debuginfo.100=SHDEBUG100_
             spv-amd-gcn-shader= spv-amd-shader-ballot= spv-amd-shader-explicit-vertex-parameter=
             spv-amd-shader-trinary-minmax= nonsemantic.clspvreflection=)
    string(REPLACE "=" ";" _v "${_v}")
    list(GET _v 0 _name)
    list(LENGTH _v _n)
    set(_prefix "")
    if(_n GREATER 1)
      list(GET _v 1 _prefix)
    endif()
    _spvt_run(${_spvt}/utils/generate_grammar_tables.py
      --extinst-vendor-grammar=${_spvt_grammar}/extinst.${_name}.grammar.json
      --vendor-insts-output=${_g}/${_name}.insts.inc
      --vendor-operand-kind-prefix=${_prefix})
  endforeach()
  foreach(_h DebugInfo=debuginfo OpenCLDebugInfo100=opencl.debuginfo.100
             NonSemanticShaderDebugInfo100=nonsemantic.shader.debuginfo.100)
    string(REPLACE "=" ";" _h "${_h}")
    list(GET _h 0 _out)
    list(GET _h 1 _name)
    _spvt_run(${_spvt}/utils/generate_language_headers.py
      --extinst-grammar=${_spvt_grammar}/extinst.${_name}.grammar.json
      --extinst-output-path=${_g}/${_out}.h)
  endforeach()
  _spvt_run(${_spvt}/utils/generate_registry_tables.py
    --xml=${_spvt_headers}/spirv/spir-v.xml --generator-output=${_g}/generators.inc)
  file(WRITE ${_g}/build-version.inc "\"${SPIRV_TOOLS_VERSION}\", \"SPIRV-Tools ${SPIRV_TOOLS_VERSION} (NDK)\"\n")
  file(WRITE ${_spvt_stamp} "${_spvt}")
  message(STATUS "SPIRV-Tools ${SPIRV_TOOLS_VERSION}: tablas generadas en ${SPIRV_TOOLS_GEN}")
endif()

# Lo que compila su Android.mk (SPIRV-Tools + SPIRV-Tools-opt): source/, util/, val/ y opt/ (sin fuzz/reduce/link/lint).
file(GLOB _spvt_sources ${_spvt}/source/*.cpp ${_spvt}/source/util/*.cpp ${_spvt}/source/val/*.cpp
     ${_spvt}/source/opt/*.cpp)
add_library(spirv_tools_opt STATIC ${_spvt_sources})
# Es código de 2022 (C++11/17): con el C++20 del resto, los operadores == reescritos dan ambigüedades.
set_target_properties(spirv_tools_opt PROPERTIES CXX_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_include_directories(spirv_tools_opt PRIVATE ${_spvt} ${_spvt_headers} ${SPIRV_TOOLS_GEN})
target_include_directories(spirv_tools_opt SYSTEM PUBLIC ${_spvt}/include)
# Sin avisos (código ajeno) y sin información de depuración: así libnnsdk.so solo crece lo que ocupa su código (~4,5 MB).
target_compile_options(spirv_tools_opt PRIVATE -w -g0)
