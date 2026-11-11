#include "loader/crash.h"

#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "loader/imports.h"

namespace {

constexpr size_t kAltStack = 256 * 1024;

bool readable(uint64_t addr) {
    // msync falla con ENOMEM si la página no está mapeada.
    return addr && msync(reinterpret_cast<void*>(addr & ~uint64_t{0xFFF}), 4096, MS_ASYNC) == 0;
}

timespec g_start;

double elapsed() {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - g_start.tv_sec) + (now.tv_nsec - g_start.tv_nsec) / 1e9;
}

std::atomic<uint64_t> g_watch_page{0};  // DEPURACIÓN: página vigilada por watch_writes

// SIGUSR2 = instantánea del watchdog: se vuelca y el juego sigue. Una escritura vigilada también se
// vuelca y sigue (se desprotege la página y la instrucción se repite). El resto, crash: se vuelca y se sale.
void on_crash(int sig, siginfo_t* info, void* ctx) {
    auto& mc = static_cast<ucontext_t*>(ctx)->uc_mcontext;
    FILE* out = stdout;
    const uint64_t page = reinterpret_cast<uint64_t>(info->si_addr) & ~uint64_t{0xFFF};
    uint64_t expected = page;
    const bool watched = sig == SIGSEGV && page && g_watch_page.compare_exchange_strong(expected, 0);
    if (watched) {
        mprotect(reinterpret_cast<void*>(page), 4096, PROT_READ | PROT_WRITE);
        fprintf(out, "\n=== ACCESO VIGILADO a %p a los %.1f s\n", info->si_addr, elapsed());
    }
    bool snapshot = sig == SIGUSR2 || watched;
    if (sig == SIGUSR2)
        fprintf(out, "\n=== INSTANTÁNEA a los %.1f s\n", elapsed());
    else if (!watched)
        fprintf(out, "\n=== CRASH a los %.1f s: %s, dirección %p\n", elapsed(), strsignal(sig), info->si_addr);
    char thread_name[32] = "?";
    pthread_getname_np(pthread_self(), thread_name, sizeof thread_name);
    fprintf(out, "hilo '%s' (tid %d)\n", thread_name, gettid());
    fprintf(out, "pc %s\nlr %s\n", describe_addr(mc.pc).c_str(), describe_addr(mc.regs[30]).c_str());
    for (int r = 0; r < 29; r++) fprintf(out, "x%-2d %016llx%s", r, mc.regs[r], r % 4 == 3 ? "\n" : "  ");
    fprintf(out, "\nfp  %016llx  sp  %016llx\n", mc.regs[29], mc.sp);
    dump_recent_stubs(out);
    fprintf(out, "Backtrace (cadena de x29):\n");
    uint64_t fp = mc.regs[29];
    for (int depth = 0; depth < 24 && fp && !(fp & 7) && readable(fp); depth++) {
        auto* frame = reinterpret_cast<uint64_t*>(fp);
        if (!frame[1]) break;
        fprintf(out, "  #%-2d %s\n", depth, describe_addr(frame[1]).c_str());
        if (frame[0] <= fp) break;
        fp = frame[0];
    }
    fflush(out);
    if (!snapshot) {
        if (g_crash_log_drain) g_crash_log_drain();
        _exit(128 + sig);
    }
}

}  // namespace

void (*g_crash_log_drain)() = nullptr;

void install_thread_altstack() {
    stack_t ss{};
    ss.ss_sp = mmap(nullptr, kAltStack, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ss.ss_size = kAltStack;
    sigaltstack(&ss, nullptr);
}

void watch_writes(uint64_t addr) {
    const uint64_t page = addr & ~uint64_t{0xFFF};
    if (mprotect(reinterpret_cast<void*>(page), 4096, PROT_NONE) == 0) {  // lecturas y escrituras
        g_watch_page = page;
        fprintf(stdout, "vigilando escrituras en la página %#llx\n", static_cast<unsigned long long>(page));
    } else {
        fprintf(stdout, "no puedo vigilar %#llx\n", static_cast<unsigned long long>(page));
    }
}

void install_crash_handler() {
    clock_gettime(CLOCK_MONOTONIC, &g_start);
    install_thread_altstack();
    struct sigaction sa{};
    sa.sa_sigaction = on_crash;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    // SIGUSR1 no sirve para el watchdog: en una app ART lo tiene bloqueado en todos los hilos.
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP, SIGABRT, SIGUSR2}) sigaction(sig, &sa, nullptr);
}
