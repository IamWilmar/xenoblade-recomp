// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Sustituto de common/logging.h de Eden: los LOG_* del recompilador van a stderr (debug y trace se
// descartan). El primer argumento (la clase de log) se ignora.

#include <cstdio>
#include <type_traits>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>

// Los enums del recompilador se registran como su valor numérico (fmt no los formatea solo).
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_enum_v<T>, char>> : fmt::formatter<std::underlying_type_t<T>> {
    auto format(T value, fmt::format_context& ctx) const {
        return fmt::formatter<std::underlying_type_t<T>>::format(static_cast<std::underlying_type_t<T>>(value), ctx);
    }
};

namespace ShimLog {

// Barreras de grupo que Eden quitó por estar dentro de control de flujo condicional (en el shader que se traduce).
inline thread_local int conditional_barriers = 0;

template <typename... Args>
void Write(const char* level, fmt::format_string<Args...> format, Args&&... args) {
    if (format.get().starts_with("Barrier within conditional")) conditional_barriers++;
    std::fprintf(stderr, "[recompilador %s] %s\n", level, fmt::format(format, std::forward<Args>(args)...).c_str());
}

}  // namespace ShimLog

#define LOG_TRACE(log_class, ...) (void(0))
#define LOG_DEBUG(log_class, ...) (void(0))
#define LOG_INFO(log_class, ...) ::ShimLog::Write("info", __VA_ARGS__)
#define LOG_WARNING(log_class, ...) ::ShimLog::Write("aviso", __VA_ARGS__)
#define LOG_ERROR(log_class, ...) ::ShimLog::Write("error", __VA_ARGS__)
#define LOG_CRITICAL(log_class, ...) ::ShimLog::Write("crítico", __VA_ARGS__)
