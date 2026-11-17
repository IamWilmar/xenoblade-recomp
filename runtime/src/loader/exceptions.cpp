// Las excepciones que lanza el juego, al log. main importa __cxa_throw; IL2CPP lanza cada excepción de C# como una
// Il2CppExceptionWrapper de C++ (un puntero al objeto System.Exception). En la consola nadie las ve (el log de Unity va
// apagado en el juego publicado), y aquí casi todas son un fallo del port: una función del SDK que contesta mal, un
// plugin que no carga, audio sin inicializar... Con el tipo, el mensaje y la pila (direcciones de main) se sabe qué
// método de C# falla sin adivinar. <datos>/sin_log_excepciones (al arrancar): directas a la de libc++abi, como antes.

#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include "loader/imports.h"

extern "C" [[noreturn]] void __cxa_throw(void* object, std::type_info* type, void (*destructor)(void*));

namespace {

// Lecturas que no tumban el proceso si el puntero no vale (los layouts de IL2CPP de abajo son los de Unity 2020.3; si
// otro juego trae otros, sale basura legible en vez de una caída).
bool safe_read(void* dst, uint64_t src, size_t size) {
    iovec local{dst, size}, remote{reinterpret_cast<void*>(src), size};
    return src && syscall(SYS_process_vm_readv, getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(size);
}
std::string c_string(uint64_t p, size_t max = 96) {
    std::string out;
    char c;
    while (out.size() < max && safe_read(&c, p + out.size(), 1) && c) out += (c >= 32 && c < 127) ? c : '?';
    return out;
}
// System.String: Il2CppObject (16 bytes), int32 length, char16 chars[].
std::string managed_string(uint64_t s, size_t max = 200) {
    int32_t length = 0;
    if (!safe_read(&length, s + 0x10, 4) || length <= 0) return {};
    std::string out;
    for (int32_t k = 0; k < length && out.size() < max; k++) {
        uint16_t c = 0;
        if (!safe_read(&c, s + 0x14 + 2 * static_cast<uint64_t>(k), 2)) break;
        if (c == '\n' || c == '\r') out += ' ';
        else if (c < 128) out += static_cast<char>(c >= 32 ? c : '?');
        else if (c < 0x800) out += static_cast<char>(0xC0 | (c >> 6)), out += static_cast<char>(0x80 | (c & 0x3F));
        else out += static_cast<char>(0xE0 | (c >> 12)), out += static_cast<char>(0x80 | ((c >> 6) & 0x3F)), out += static_cast<char>(0x80 | (c & 0x3F));
    }
    return out;
}
// Il2CppObject: klass en +0. Il2CppClass: image, gc_desc, name (+0x10), namespaze (+0x18).
std::string class_name(uint64_t object) {
    uint64_t klass = 0, name = 0, space = 0;
    if (!safe_read(&klass, object, 8) || !safe_read(&name, klass + 0x10, 8) || !safe_read(&space, klass + 0x18, 8)) return "?";
    const std::string ns = c_string(space), n = c_string(name);
    return ns.empty() ? n : ns + "." + n;
}

struct Seen {
    uint64_t count = 0;
};
std::mutex g_mutex;
std::unordered_map<uint64_t, Seen> g_seen;  // (sitio que lanza, clase) -> veces
std::atomic<bool> g_enabled{true};

void log_throw(void* object, std::type_info* type, uint64_t lr, const uint64_t* frame) {
    uint64_t type_name_ptr = 0;
    safe_read(&type_name_ptr, reinterpret_cast<uint64_t>(type) + 8, 8);  // std::type_info: vtable, nombre
    const std::string cxx_type = c_string(type_name_ptr);
    uint64_t exception = 0;
    std::string what = cxx_type, message;
    uint64_t klass = 0;
    if (cxx_type.find("Il2CppExceptionWrapper") != std::string::npos && safe_read(&exception, reinterpret_cast<uint64_t>(object), 8)) {
        what = class_name(exception);
        safe_read(&klass, exception, 8);
        uint64_t text = 0;  // Il2CppException: object (16), className, message (+0x18)
        if (safe_read(&text, exception + 0x18, 8)) message = managed_string(text);
    }
    uint64_t n;
    {
        std::lock_guard lock(g_mutex);
        n = ++g_seen[lr * 1000003u ^ klass ^ std::hash<std::string>{}(cxx_type)].count;
        if (g_seen.size() > 4096) g_seen.clear();
    }
    // La primera vez, con su pila; luego en las potencias de dos (las que se repiten cada frame no llenan el log).
    if (n > 1 && (n & (n - 1)) != 0) return;
    std::string stack;
    const uint64_t* fp = frame;
    for (int k = 0; k < 14 && fp; k++) {
        uint64_t pair[2];
        if (!safe_read(pair, reinterpret_cast<uint64_t>(fp), 16) || !pair[1]) break;
        stack += " " + describe_addr(pair[1] - 4);
        if (pair[0] <= reinterpret_cast<uint64_t>(fp)) break;  // la cadena de marcos solo sube
        fp = reinterpret_cast<const uint64_t*>(pair[0]);
    }
    printf("[excepción] %s%s%s%s (x%llu) desde %s; pila:%s\n", what.c_str(), message.empty() ? "" : ": \"", message.c_str(),
           message.empty() ? "" : "\"", static_cast<unsigned long long>(n), describe_addr(lr - 4).c_str(), stack.c_str());
    fflush(stdout);
}

}  // namespace

extern "C" [[noreturn]] void nier_cxa_throw(void* object, std::type_info* type, void (*destructor)(void*)) {
    static const bool enabled = [] {
        const char* data = getenv("NIER_DATA");
        return access((std::string(data ? data : ".") + "/sin_log_excepciones").c_str(), F_OK) != 0;
    }();
    if (enabled)
        log_throw(object, type, reinterpret_cast<uint64_t>(__builtin_return_address(0)),
                  static_cast<const uint64_t*>(__builtin_frame_address(0)));
    __cxa_throw(object, type, destructor);
}
