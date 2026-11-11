// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Sustituto de common/assert.h de Eden: mismos macros. Una aserción fallida lanza una excepción en vez
// de abortar: nvnshader::translate la recoge y ese shader se da por no traducible (en la Flip, abortar
// mataría el juego).

#include <stdexcept>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"

namespace nvnshader_shim {
template <typename... Args>
[[noreturn]] inline void fail(const char* what, const char* file, int line, fmt::format_string<Args...> f, Args&&... args) {
    throw std::runtime_error(std::string(what) + " (" + file + ":" + std::to_string(line) + "): " +
                             fmt::format(f, std::forward<Args>(args)...));
}
}  // namespace nvnshader_shim

#define ASSERT_MSG(_a_, ...)                                                                       \
    do {                                                                                           \
        if (!(_a_)) [[unlikely]] {                                                                 \
            ::nvnshader_shim::fail("aserción", __FILE__, __LINE__, __VA_ARGS__);                     \
        }                                                                                          \
    } while (0)
#define ASSERT(_a_) ASSERT_MSG(_a_, "{}", #_a_)
#define UNREACHABLE_MSG(...)                                                                       \
    do {                                                                                           \
        ::nvnshader_shim::fail("inalcanzable", __FILE__, __LINE__, __VA_ARGS__);                     \
    } while (0)
#define UNREACHABLE() UNREACHABLE_MSG("{}", "")
#define DEBUG_ASSERT(_a_) ASSERT(_a_)
#define DEBUG_ASSERT_MSG(_a_, ...) ASSERT_MSG(_a_, __VA_ARGS__)
#define UNIMPLEMENTED() ASSERT(false && "Unimplemented!")
#define UNIMPLEMENTED_MSG(...) ASSERT_MSG(false, __VA_ARGS__)
#define UNIMPLEMENTED_IF(cond) ASSERT((!(cond)) && "Unimplemented!")
#define UNIMPLEMENTED_IF_MSG(cond, ...) ASSERT_MSG(!(cond), __VA_ARGS__)
#define ASSERT_OR_EXECUTE_MSG(_a_, _b_, ...)                                                       \
    do {                                                                                           \
        if (!(_a_)) [[unlikely]] {                                                                 \
            LOG_CRITICAL(Debug, __VA_ARGS__);                                                      \
            _b_                                                                                    \
        }                                                                                          \
    } while (0)
#define ASSERT_OR_EXECUTE(_a_, _b_) ASSERT_OR_EXECUTE_MSG(_a_, _b_, "{}", #_a_)
