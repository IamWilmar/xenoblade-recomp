// SPDX-License-Identifier: GPL-3.0-or-later
// Prueba en el Mac de la contabilidad de la reproducción en diferido (commands.h): compila texture.cpp y cmdbuf.cpp TAL
// CUAL (más layout.cpp y formats.cpp) y pone en lugar del ejecutor una cola falsa con las mismas reglas de tickets que
// SubmitQueue (executor.cpp: fake_submit y fake_replay copian su bucle). No prueba Vulkan ni la reproducción: solo quién
// cuenta qué (Texture::open_refs y last_ticket, las listas llamadas con CallCommands), qué se queda una textura al
// entrar en el limbo (sus texels, sus cajas de WriteTexels, la clave de sus alias) y cuándo sale (reap_limbo_textures y sus
// válvulas). Con ASan y UBSan: también que nada lea memoria que el juego ya soltó ni objetos borrados.
//
//   tools/tests/diferido_test.sh            compila y corre los cuatro modos (los interruptores en vivo de cada uno)
//   BASE=8b8e464 tools/tests/diferido_test.sh   además, el código de ese commit prueba a prueba (qué fallaba antes)
//
// A mano: diferido_test <carpeta de interruptores> [normal|sin_refs|sin_valvulas|recientes] [una prueba]
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <variant>
#include <vector>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/formats.h"

using namespace nvn;

// ---------------------------------------------------------------------------------------------------------------------
// Lo que dan executor.cpp, memory.cpp, program.cpp, queue.cpp y procs.cpp.
namespace nvn {
void register_proc(const char*, void*) {}
void* find_proc(const char*) { return nullptr; }
Objects<MemoryPool> g_memory_pools;
Objects<Window> g_windows;

namespace {
std::recursive_mutex g_mutex;
uint64_t g_queued = 0, g_done = 0;
std::atomic<uint32_t> g_game_frame{0};
constexpr uint32_t kFrameRing = 64;
std::atomic<uint64_t> g_frame_end_ticket[kFrameRing];
int g_retired = 0, g_barriers = 0;
}  // namespace

// El bucle de SubmitQueue::submit sobre las texturas de cada lista (executor.cpp), sin la cola.
uint64_t fake_submit(std::initializer_list<CommandList*> lists) {
    const uint64_t ticket = ++g_queued;
    for (CommandList* l : lists) {
        for (Texture* t : l->texture_refs) {
            __atomic_store_n(&t->last_ticket, ticket, __ATOMIC_RELEASE);
            if (!l->submitted) __atomic_fetch_sub(&t->open_refs, 1, __ATOMIC_RELEASE);
        }
        l->submitted = true;
    }
    return ticket;
}
// SubmitQueue::run tras reproducir un envío.
void fake_replay(uint64_t ticket) {
    g_done = std::max(g_done, ticket);
    reap_limbo_textures(g_done);
}
uint64_t last_submit() { return g_queued; }
uint64_t replayed_ticket() { return g_done; }
uint32_t game_frame() { return g_game_frame.load(std::memory_order_relaxed); }
void note_game_present() {
    const uint32_t frame = g_game_frame.load(std::memory_order_relaxed);
    g_frame_end_ticket[frame % kFrameRing].store(g_queued, std::memory_order_release);
    g_game_frame.store(frame + 1, std::memory_order_release);
}
bool game_frame_replayed(uint32_t frame, uint64_t replayed) {
    const uint32_t now = g_game_frame.load(std::memory_order_acquire);
    if (now == frame) return false;
    if (now - frame > kFrameRing) return true;
    return replayed >= g_frame_end_ticket[frame % kFrameRing].load(std::memory_order_acquire);
}
void note_list_dropped(CommandList& list) {
    if (!list.submitted)
        for (Texture* t : list.texture_refs) __atomic_fetch_sub(&t->open_refs, 1, __ATOMIC_RELEASE);
    list.texture_refs.clear();
    list.submitted = false;
}
void replay_barrier(uint64_t, const char*) { g_barriers++; }
std::recursive_mutex& executor_mutex() { return g_mutex; }
void note_executor_lock_wait(uint64_t) {}
void note_executor_lock_waiting(int) {}
void retire_texture_gpu(Texture& t) {
    g_retired++;
    t.image = VK_NULL_HANDLE;
    t.memory = {};
    t.gpu_bytes = 0;
}
void retire_sampler(VkSampler) {}
void release_command_list(CommandList* list) { recycle_command_list(list); }
Program* find_program(const void*) { return nullptr; }
void snapshot_program(Program&) {}
uint32_t* event_value(const void*) { return nullptr; }
}  // namespace nvn

// ---------------------------------------------------------------------------------------------------------------------
// Las entradas de NVN que usa la prueba (texture.cpp, cmdbuf.cpp).
extern "C" {
void nvnTextureBuilderSetDefaults(TextureDesc* b);
void nvnTextureBuilderSetSize2D(TextureDesc* b, int w, int h);
void nvnTextureBuilderSetFormat(TextureDesc* b, int format);
void nvnTextureBuilderSetStorage(TextureDesc* b, void* pool, int64_t offset);
size_t nvnTextureBuilderGetStorageSize(const TextureDesc* b);
Boolean nvnTextureInitialize(void* texture, const TextureDesc* b);
void nvnTextureFinalize(void* texture);
void nvnTextureWriteTexels(const void* t, const TextureViewDesc* view, const CopyRegion* region, const void* data);
Boolean nvnTexturePoolInitialize(void* pool, void* memory_pool, int64_t offset, int count);
void nvnTexturePoolFinalize(void* pool);
void nvnTexturePoolRegisterTexture(void* pool, int id, const void* texture, const TextureViewDesc* view);
Boolean nvnCommandBufferInitialize(void* cmd, void*);
void nvnCommandBufferFinalize(void* cmd);
void nvnCommandBufferBeginRecording(void* cmd);
uint64_t nvnCommandBufferEndRecording(void* cmd);
void nvnCommandBufferSetRenderTargets(void* cmd, int count, void* const* colors, const void* const* color_views, void* depth,
                                      const void* depth_view);
void nvnCommandBufferCopyTextureToTexture(void* cmd, const void* src, const void* src_view, const CopyRegion* src_region,
                                          const void* dst, const void* dst_view, const CopyRegion* dst_region, int);
void nvnCommandBufferCopyBufferToTexture(void* cmd, uint64_t src, const void* dst, const void* dst_view,
                                         const CopyRegion* region, int);
void nvnCommandBufferCallCommands(void* cmd, int count, const uint64_t* handles);
void nvnCommandBufferSetTexturePool(void* cmd, const void* pool);
}

// ---------------------------------------------------------------------------------------------------------------------
#ifdef BASE  // el código de antes (commit 8b8e464): sin Texture::limbo, sin estadísticas del limbo ni huella de subidas
namespace nvn {
void take_limbo_stats(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t& e, uint64_t& f) { a = b = c = d = e = f = 0; }
}  // namespace nvn
#define IN_LIMBO(t) true
#define NOT_LIMBO(t) true
#define STAT(cond) true
#else
#define IN_LIMBO(t) ((t)->limbo)
#define NOT_LIMBO(t) (!(t)->limbo)
#define STAT(cond) (cond)
#endif

namespace {
int g_failed = 0, g_checked = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        g_checked++;                                                                  \
        if (!(cond)) {                                                                \
            g_failed++;                                                               \
            printf("FALLA %s:%d: %s\n", __func__, __LINE__, #cond);                   \
        }                                                                             \
    } while (0)

constexpr int kRGBA8 = 0x25;

// Un objeto del juego: memoria opaca con sitio para la marca que deja nvnTextureInitialize.
struct GameObject {
    alignas(16) uint8_t bytes[256] = {};
};

// Un memory pool del juego con su memoria.
struct Pool {
    GameObject object;
    std::vector<uint8_t> memory;
    explicit Pool(size_t size) : memory(size) {
        MemoryPool& p = g_memory_pools.create(&object);
        p.storage = memory.data();
        p.size = size;
    }
    void finalize() { g_memory_pools.destroy(&object); }
    ~Pool() { finalize(); }
};

Texture* make_texture(GameObject& object, Pool& pool, int64_t offset, int w, int h) {
    TextureDesc b;
    nvnTextureBuilderSetDefaults(&b);
    nvnTextureBuilderSetSize2D(&b, w, h);
    nvnTextureBuilderSetFormat(&b, kRGBA8);
    nvnTextureBuilderSetStorage(&b, &pool.object, offset);
    nvnTextureInitialize(&object, &b);
    return g_textures.get(&object);
}
// Lo que haría el ejecutor al usarla: una imagen (de mentira) ya subida.
void fake_image(Texture* t, uint64_t gpu_bytes) {
    t->image = reinterpret_cast<VkImage>(uintptr_t{0x1000} + t->serial);
    t->gpu_bytes = gpu_bytes;
    t->uploaded = true;
}
CommandList* list_of(uint64_t handle) { return reinterpret_cast<CommandList*>(handle); }
int open_refs(const Texture* t) { return __atomic_load_n(&t->open_refs, __ATOMIC_ACQUIRE); }
// Un frame del juego sin nada: presenta y la cola reproduce lo que hubiera.
void idle_frame() {
    note_game_present();
    fake_replay(g_queued);
}
std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t k = 0; k < n; k++) v[k] = static_cast<uint8_t>(seed + k * 7);
    return v;
}

// La textura temporal de una página de AmplifyTexture: WriteTexels, copia grabada y enviada, y el juego la finaliza (y
// reutiliza su memoria) con la copia aún sin reproducir.
void test_copia_pendiente() {
    Pool pool(1 << 20);
    GameObject a_obj, b_obj, cb;
    Texture* a = make_texture(a_obj, pool, 0, 136, 136);
    Texture* b = make_texture(b_obj, pool, 512 << 10, 136, 136);
    const size_t bytes = 136 * 136 * 4;
    const std::vector<uint8_t> p1 = pattern(bytes, 3), p2 = pattern(bytes, 91);
    const CopyRegion all{0, 0, 0, 136, 136, 1};
    nvnTextureWriteTexels(&a_obj, nullptr, &all, p1.data());
    std::vector<uint8_t> swizzled(pool.memory.begin(), pool.memory.begin() + a->layout.total_size());

    nvnCommandBufferInitialize(&cb, nullptr);
    nvnCommandBufferBeginRecording(&cb);
    nvnCommandBufferCopyTextureToTexture(&cb, &a_obj, nullptr, &all, &b_obj, nullptr, &all, 0);
    CommandList* l = list_of(nvnCommandBufferEndRecording(&cb));
    CHECK(open_refs(a) == 1 && open_refs(b) == 1);
    const uint64_t ticket = fake_submit({l});
    CHECK(open_refs(a) == 0 && a->last_ticket == ticket);
    note_game_present();  // el juego pasa al frame siguiente; la cola aún no ha reproducido nada

    const size_t limbo_before = limbo_textures();
    nvnTextureFinalize(&a_obj);
    CHECK(limbo_textures() == limbo_before + 1);
    CHECK(IN_LIMBO(a) && a->storage && a->storage != pool.memory.data());
    nvnTextureWriteTexels(&b_obj, nullptr, &all, p2.data());             // otra cosa...
    memset(pool.memory.data(), 0xEE, a->layout.total_size());           // ...y la memoria de A, reutilizada
    CHECK(a->storage && memcmp(a->storage, swizzled.data(), swizzled.size()) == 0);  // la copia tiene sus texels
    CHECK(!a->texel_writes.empty());                                     // y sus cajas siguen para subirse

    fake_replay(ticket);  // la cola reproduce la copia, pero el frame en que se finalizó no ha acabado
    CHECK(limbo_textures() == limbo_before + 1);
    idle_frame();
    CHECK(limbo_textures() == limbo_before);
    CHECK(IN_LIMBO(a) && !a->storage && a->vk_failed);
    nvnTextureFinalize(&b_obj);
    idle_frame();
    nvnCommandBufferFinalize(&cb);
}

// La copia está grabada en una lista que aún no se ha enviado cuando el juego finaliza el origen.
void test_copia_sin_enviar() {
    Pool pool(1 << 20);
    GameObject a_obj, b_obj, cb;
    Texture* a = make_texture(a_obj, pool, 0, 64, 64);
    make_texture(b_obj, pool, 512 << 10, 64, 64);
    const CopyRegion all{0, 0, 0, 64, 64, 1};
    const std::vector<uint8_t> p1 = pattern(64 * 64 * 4, 11);
    nvnTextureWriteTexels(&a_obj, nullptr, &all, p1.data());
    nvnCommandBufferInitialize(&cb, nullptr);
    nvnCommandBufferBeginRecording(&cb);
    nvnCommandBufferCopyTextureToTexture(&cb, &a_obj, nullptr, &all, &b_obj, nullptr, &all, 0);
    const size_t before = limbo_textures();
    nvnTextureFinalize(&a_obj);  // con la lista aún grabándose
    CHECK(limbo_textures() == before + 1 && IN_LIMBO(a) && a->storage);
    idle_frame();
    idle_frame();
    CHECK(limbo_textures() == before + 1);  // sigue: la lista no se ha enviado
    CommandList* l = list_of(nvnCommandBufferEndRecording(&cb));
    const uint64_t ticket = fake_submit({l});
    fake_replay(ticket - 1);
    CHECK(limbo_textures() == before + 1);  // enviada pero sin reproducir
    idle_frame();
    CHECK(limbo_textures() == before);
    nvnTextureFinalize(&b_obj);
    idle_frame();
    nvnCommandBufferFinalize(&cb);
}

// Graphics jobs de Unity: el job graba SetRenderTargets en la lista de su hilo y el principal la llama.
void test_llamadas(bool transfer) {
    Pool pool(4 << 20);
    GameObject rt_obj, job, main_cb;
    Texture* rt = make_texture(rt_obj, pool, 0, 256, 256);
    nvnCommandBufferInitialize(&job, nullptr);
    nvnCommandBufferInitialize(&main_cb, nullptr);
    nvnCommandBufferBeginRecording(&job);
    void* colors[1] = {&rt_obj};
    nvnCommandBufferSetRenderTargets(&job, 1, colors, nullptr, nullptr, nullptr);
    const uint64_t called = nvnCommandBufferEndRecording(&job);
    CHECK(open_refs(rt) == 1);
    nvnCommandBufferBeginRecording(&main_cb);
    nvnCommandBufferCallCommands(&main_cb, 1, &called);
    CommandList* m = list_of(nvnCommandBufferEndRecording(&main_cb));
    CHECK(open_refs(rt) == 1);
    CHECK(m->texture_refs.size() == (transfer ? 1u : 0u));
    CHECK(list_of(called)->submitted == transfer);
    CHECK(m->commands.size() == 1 && std::holds_alternative<cmd::SetRenderTargets>(m->commands[0]));

    const uint64_t ticket = fake_submit({m});
    if (transfer) {
        CHECK(open_refs(rt) == 0 && rt->last_ticket == ticket);  // quien la lleva a la cola le pone su ticket
    } else {
        CHECK(open_refs(rt) == 1 && rt->last_ticket == 0);       // como antes: ni ticket ni deja de contarla
    }
    // El juego la finaliza en el mismo frame, con el envío sin reproducir: al limbo en los dos casos.
    const size_t before = limbo_textures();
    fake_image(rt, 1 << 20);
    rt->gpu_written = true;
    nvnTextureFinalize(&rt_obj);
    CHECK(limbo_textures() == before + 1);
    note_game_present();
    fake_replay(ticket);
    if (transfer) {
        CHECK(limbo_textures() == before);      // reproducido su envío y acabado el frame: fuera
    } else {
        CHECK(limbo_textures() == before + 1);  // la lista llamada la sigue contando hasta reciclarse
    }
    nvnCommandBufferFinalize(&job);  // el juego recicla la lista llamada
    CHECK(open_refs(rt) == 0);       // nunca negativa
    idle_frame();
    CHECK(limbo_textures() == before);
    nvnCommandBufferFinalize(&main_cb);
    CHECK(open_refs(rt) == 0);
}

// La misma lista llamada dos veces y además enviada: las cuentas cuadran.
void test_llamada_doble() {
    Pool pool(4 << 20);
    GameObject rt_obj, job, a_cb, b_cb;
    Texture* rt = make_texture(rt_obj, pool, 0, 64, 64);
    nvnCommandBufferInitialize(&job, nullptr);
    nvnCommandBufferInitialize(&a_cb, nullptr);
    nvnCommandBufferInitialize(&b_cb, nullptr);
    nvnCommandBufferBeginRecording(&job);
    void* colors[1] = {&rt_obj};
    nvnCommandBufferSetRenderTargets(&job, 1, colors, nullptr, nullptr, nullptr);
    const uint64_t called = nvnCommandBufferEndRecording(&job);
    nvnCommandBufferBeginRecording(&a_cb);
    nvnCommandBufferCallCommands(&a_cb, 1, &called);
    CommandList* a = list_of(nvnCommandBufferEndRecording(&a_cb));
    nvnCommandBufferBeginRecording(&b_cb);
    nvnCommandBufferCallCommands(&b_cb, 1, &called);
    CommandList* b = list_of(nvnCommandBufferEndRecording(&b_cb));
    CHECK(open_refs(rt) == 2);
    fake_submit({a});
    CHECK(open_refs(rt) == 1);
    const uint64_t direct = fake_submit({list_of(called)});
    CHECK(open_refs(rt) == 1 && rt->last_ticket == direct);
    nvnCommandBufferFinalize(&b_cb);  // b nunca se envía: su mención se va con ella
    CHECK(open_refs(rt) == 0);
    (void)b;
    nvnCommandBufferFinalize(&a_cb);
    nvnCommandBufferFinalize(&job);
    CHECK(open_refs(rt) == 0);
    fake_replay(g_queued);
    nvnTextureFinalize(&rt_obj);
    idle_frame();
}

// nvnTextureInitialize sobre un objeto vivo con una lista sin enviar que lo nombra.
void test_reinicio() {
    Pool pool(1 << 20);
    GameObject x_obj, y_obj, cb;
    Texture* x = make_texture(x_obj, pool, 0, 32, 32);
    make_texture(y_obj, pool, 256 << 10, 32, 32);
    const CopyRegion all{0, 0, 0, 32, 32, 1};
    nvnCommandBufferInitialize(&cb, nullptr);
    nvnCommandBufferBeginRecording(&cb);
    nvnCommandBufferCopyTextureToTexture(&cb, &x_obj, nullptr, &all, &y_obj, nullptr, &all, 0);
    CommandList* l = list_of(nvnCommandBufferEndRecording(&cb));
    CHECK(open_refs(x) == 1);
    Texture* again = make_texture(x_obj, pool, 0, 32, 32);
    CHECK(again == x);            // el mismo objeto
    CHECK(open_refs(x) == 1);     // y la lista sin enviar lo sigue contando
    const uint64_t ticket = fake_submit({l});
    CHECK(open_refs(x) == 0 && x->last_ticket == ticket);
    fake_replay(ticket);
    nvnTextureFinalize(&x_obj);
    nvnTextureFinalize(&y_obj);
    idle_frame();
    nvnCommandBufferFinalize(&cb);
}

// El pool (y su memoria) se fue antes que la textura: no se copia de memoria que ya no es del juego.
void test_pool_finalizado() {
    auto* pool = new Pool(1 << 20);
    GameObject a_obj, b_obj, cb;
    Texture* a = make_texture(a_obj, *pool, 0, 64, 64);
    Pool other(1 << 20);
    make_texture(b_obj, other, 0, 64, 64);
    const CopyRegion all{0, 0, 0, 64, 64, 1};
    const std::vector<uint8_t> p = pattern(64 * 64 * 4, 5);
    nvnTextureWriteTexels(&a_obj, nullptr, &all, p.data());
    nvnCommandBufferInitialize(&cb, nullptr);
    nvnCommandBufferBeginRecording(&cb);
    nvnCommandBufferCopyTextureToTexture(&cb, &a_obj, nullptr, &all, &b_obj, nullptr, &all, 0);
    CommandList* l = list_of(nvnCommandBufferEndRecording(&cb));
    const uint64_t ticket = fake_submit({l});
    pool->finalize();
    delete pool;  // con ASan: leer su memoria desde aquí es un error
    const size_t before = limbo_textures();
    nvnTextureFinalize(&a_obj);
    CHECK(limbo_textures() == before + 1);
    CHECK(IN_LIMBO(a) && !a->storage && a->texel_writes.empty());
    note_game_present();
    fake_replay(ticket);
    CHECK(limbo_textures() == before);
    nvnTextureFinalize(&b_obj);
    idle_frame();
    nvnCommandBufferFinalize(&cb);
}

// Válvula de edad: una lista que nombra la textura y que el juego ni envía ni recicla nunca.
void test_valvula_edad(bool valves) {
    Pool pool(1 << 20);
    GameObject a_obj, b_obj, cb;
    Texture* a = make_texture(a_obj, pool, 0, 32, 32);
    make_texture(b_obj, pool, 256 << 10, 32, 32);
    const CopyRegion all{0, 0, 0, 32, 32, 1};
    nvnCommandBufferInitialize(&cb, nullptr);
    nvnCommandBufferBeginRecording(&cb);
    nvnCommandBufferCopyTextureToTexture(&cb, &a_obj, nullptr, &all, &b_obj, nullptr, &all, 0);
    nvnCommandBufferEndRecording(&cb);  // grabada y olvidada
    fake_image(a, 4096);
    const size_t before = limbo_textures();
    nvnTextureFinalize(&a_obj);
    CHECK(limbo_textures() == before + 1);
    // reap solo corre tras reproducir un envío: uno vacío por frame.
    GameObject cb2;
    nvnCommandBufferInitialize(&cb2, nullptr);
    auto frame = [&] {
        nvnCommandBufferBeginRecording(&cb2);
        CommandList* l = list_of(nvnCommandBufferEndRecording(&cb2));
        const uint64_t t = fake_submit({l});
        note_game_present();
        fake_replay(t);
    };
    for (int k = 0; k < 16; k++) frame();
    CHECK(limbo_textures() == before + 1);  // 16 frames: aún no
    frame();
    CHECK(limbo_textures() == (valves ? before : before + 1));
    uint64_t max_count, max_bytes, forced_age, forced_budget, young, kept;
    take_limbo_stats(max_count, max_bytes, forced_age, forced_budget, young, kept);
    CHECK(STAT(forced_age == (valves ? 1u : 0u) && forced_budget == 0));
    nvnCommandBufferFinalize(&cb);  // ahora sí la recicla el juego
    CHECK(open_refs(a) == 0);
    frame();
    CHECK(limbo_textures() == before);
    nvnTextureFinalize(&b_obj);
    idle_frame();
    nvnCommandBufferFinalize(&cb2);
}

// Válvula de tope: el juego suelta imágenes grandes sin presentar (una carga).
void test_valvula_tope(bool valves) {
    Pool pool(8 << 20);
    GameObject objs[5], cb;
    Texture* t[5];
    const size_t before = limbo_textures();
    nvnCommandBufferInitialize(&cb, nullptr);
    auto submit_empty = [&] {
        nvnCommandBufferBeginRecording(&cb);
        return fake_submit({list_of(nvnCommandBufferEndRecording(&cb))});
    };
    for (int k = 0; k < 5; k++) {
        t[k] = make_texture(objs[k], pool, int64_t{k} << 20, 64, 64);
        fake_image(t[k], uint64_t{100} << 20);
        nvnTextureFinalize(&objs[k]);
    }
    CHECK(limbo_textures() == before + 5);
    fake_replay(g_queued);  // nada nuevo reproducido tras finalizarlas: ninguna sale aún
    CHECK(limbo_textures() == before + 5);
    const uint64_t ticket = submit_empty();  // el vaciado siguiente del juego (la lista que se grababa)
    fake_replay(ticket);
    // 500 MB > 256 MB: salen las más antiguas hasta quedar por debajo (quedan 2 = 200 MB). Sin válvulas, todas siguen.
    CHECK(limbo_textures() == (valves ? before + 2 : before + 5));
    uint64_t max_count, max_bytes, forced_age, forced_budget, young, kept;
    take_limbo_stats(max_count, max_bytes, forced_age, forced_budget, young, kept);
    CHECK(STAT(forced_budget == (valves ? 3u : 0u)));
    CHECK(STAT(max_bytes == uint64_t{500} << 20 && max_count >= 5));
    idle_frame();  // por fin presenta: el resto
    CHECK(limbo_textures() == before);
    for (Texture* x : t) CHECK(!x->image && x->vk_failed);
    nvnCommandBufferFinalize(&cb);
}

// Recién creada y finalizada sin imagen ni comandos que la nombren.
void test_recientes(bool keep) {
    Pool pool(1 << 20);
    GameObject a_obj, old_obj;
    Texture* old = make_texture(old_obj, pool, 512 << 10, 64, 64);
    idle_frame();
    idle_frame();
    Texture* a = make_texture(a_obj, pool, 0, 64, 64);
    const CopyRegion all{0, 0, 0, 64, 64, 1};
    const std::vector<uint8_t> p = pattern(64 * 64 * 4, 77);
    nvnTextureWriteTexels(&a_obj, nullptr, &all, p.data());
    std::vector<uint8_t> swizzled(pool.memory.begin(), pool.memory.begin() + a->layout.total_size());
    uint64_t max_count, max_bytes, forced_age, forced_budget, young, kept;
    take_limbo_stats(max_count, max_bytes, forced_age, forced_budget, young, kept);
    const size_t before = limbo_textures();
    nvnTextureFinalize(&a_obj);
    nvnTextureFinalize(&old_obj);  // creada hace 2 frames: no cuenta como reciente
    take_limbo_stats(max_count, max_bytes, forced_age, forced_budget, young, kept);
    CHECK(STAT(young == 1));
    CHECK(NOT_LIMBO(old));
    if (keep) {
        CHECK(limbo_textures() == before + 1 && IN_LIMBO(a) && a->storage);
        CHECK(a->storage != pool.memory.data() && memcmp(a->storage, swizzled.data(), swizzled.size()) == 0);
        idle_frame();
        CHECK(limbo_textures() == before);
    } else {
        CHECK(limbo_textures() == before && NOT_LIMBO(a) && !a->storage);
    }
}

// Con imagen, sin comandos que la nombren y con un WriteTexels aún sin subir: los draws por id esperan verlo.
void test_cajas_sin_subir() {
    Pool pool(1 << 20);
    GameObject a_obj;
    Texture* a = make_texture(a_obj, pool, 0, 64, 64);
    idle_frame();
    idle_frame();
    fake_image(a, 1 << 16);
    const CopyRegion all{0, 0, 0, 64, 64, 1};
    const std::vector<uint8_t> p = pattern(64 * 64 * 4, 200);
    nvnTextureWriteTexels(&a_obj, nullptr, &all, p.data());
    std::vector<uint8_t> swizzled(pool.memory.begin(), pool.memory.begin() + a->layout.total_size());
    const size_t before = limbo_textures();
    nvnTextureFinalize(&a_obj);
    CHECK(limbo_textures() == before + 1);
    memset(pool.memory.data(), 0, a->layout.total_size());
    CHECK(a->storage && a->storage != pool.memory.data() && memcmp(a->storage, swizzled.data(), swizzled.size()) == 0);
    CHECK(!a->texel_writes.empty());
    idle_frame();
    CHECK(limbo_textures() == before && a->texel_writes.empty());
}

// Render target ya pintado: su memoria se deja como clave de sus alias; no se copia ni se lee.
void test_render_target() {
    Pool pool(1 << 20);
    GameObject a_obj;
    Texture* a = make_texture(a_obj, pool, 0, 64, 64);
    fake_image(a, 1 << 16);
    a->gpu_written = true;
    uint64_t total, pending, unsent, limbo, copied, after[5];
    take_teardown_stats(total, pending, unsent, limbo, copied, after);
    const size_t before = limbo_textures();
    nvnTextureFinalize(&a_obj);
    take_teardown_stats(total, pending, unsent, limbo, copied, after);
    CHECK(limbo_textures() == before + 1 && limbo == 1 && copied == 0);
    CHECK(IN_LIMBO(a) && a->storage == pool.memory.data());
    idle_frame();
    CHECK(limbo_textures() == before && !a->storage);
}

// La huella de los extremos del buffer de una subida, al grabarla.
#ifndef BASE
void test_huella_subida() {
    Pool pool(1 << 20);
    GameObject a_obj, cb;
    make_texture(a_obj, pool, 0, 64, 32);
    const CopyRegion all{0, 0, 0, 64, 32, 1};
    std::vector<uint8_t> staging = pattern(64 * 32 * 4, 9);
    nvnCommandBufferInitialize(&cb, nullptr);
    nvnCommandBufferBeginRecording(&cb);
    nvnCommandBufferCopyBufferToTexture(&cb, reinterpret_cast<uint64_t>(staging.data()), &a_obj, nullptr, &all, 0);
    CommandList* l = list_of(nvnCommandBufferEndRecording(&cb));
    CHECK(l->commands.size() == 1);
    const auto* c = std::get_if<cmd::CopyBufferToTexture>(&l->commands[0]);
    CHECK(c && c->span == staging.size());
    CHECK(c && c->print == cmd::edge_print(staging.data(), staging.size()));
    staging[staging.size() - 1] ^= 0x40;  // el juego reutiliza el buffer: el último byte ya cambia la huella
    CHECK(c && c->print != cmd::edge_print(staging.data(), staging.size()));
    staging[staging.size() - 1] ^= 0x40;
    staging[1000] ^= 0x40;  // el centro no entra en la huella (solo los extremos)
    CHECK(c && c->print == cmd::edge_print(staging.data(), staging.size()));
    CHECK(cmd::edge_print(staging.data(), 10) != cmd::edge_print(staging.data(), 11));
    fake_replay(fake_submit({l}));
    nvnTextureFinalize(&a_obj);
    idle_frame();
    nvnCommandBufferFinalize(&cb);
}
#endif

// Un pool de texturas finalizado con un SetPools grabado que aún lo nombra: el objeto sigue (con ASan, leerlo liberado
// es un error).
void test_pool_de_texturas() {
    Pool pool(1 << 20);
    GameObject tp, cb, a_obj;
    Texture* a = make_texture(a_obj, pool, 0, 16, 16);
    nvnTexturePoolInitialize(&tp, &pool.object, 0, 64);
    nvnTexturePoolRegisterTexture(&tp, 5, &a_obj, nullptr);
    nvnCommandBufferInitialize(&cb, nullptr);
    nvnCommandBufferBeginRecording(&cb);
    nvnCommandBufferSetTexturePool(&cb, &tp);
    CommandList* l = list_of(nvnCommandBufferEndRecording(&cb));
    const auto* c = std::get_if<cmd::SetPools>(&l->commands[0]);
    CHECK(c && c->textures);
    nvnTexturePoolFinalize(&tp);
    CHECK(!g_texture_pools.get(&tp));
    CHECK(c && c->textures->textures.size() == 64 && c->textures->textures[5].texture == a);
    fake_replay(fake_submit({l}));
    nvnTextureFinalize(&a_obj);
    idle_frame();
    nvnCommandBufferFinalize(&cb);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const std::string dir = argv[1], mode = argc > 2 ? argv[2] : "normal";
    setenv("NIER_DATA", dir.c_str(), 1);
    auto flag = [&](const char* name) {
        if (FILE* f = fopen((dir + "/" + name).c_str(), "w")) fclose(f);
    };
    for (const char* name : {"sin_refs_llamadas", "limbo_sin_valvulas", "limbo_recientes"}) remove((dir + "/" + name).c_str());
    if (mode == "sin_refs") flag("sin_refs_llamadas");
    if (mode == "sin_valvulas") flag("limbo_sin_valvulas");
    if (mode == "recientes") flag("limbo_recientes");
    const bool transfer = mode != "sin_refs", valves = mode != "sin_valvulas", recent = mode == "recientes";
    setvbuf(stdout, nullptr, _IOLBF, 0);

    const std::string only = argc > 3 ? argv[3] : "";
    auto run = [&](const char* name, auto&& fn) {
        if (!only.empty() && only != name) return;
        const int failed = g_failed;
        fn();
        if (!only.empty() || g_failed != failed) printf("%s: %s\n", name, g_failed == failed ? "bien" : "FALLA");
    };
    run("copia_pendiente", [] { test_copia_pendiente(); });
    run("copia_sin_enviar", [] { test_copia_sin_enviar(); });
    run("llamadas", [&] { test_llamadas(transfer); });
    if (transfer) run("llamada_doble", [] { test_llamada_doble(); });
    run("reinicio", [] { test_reinicio(); });
    run("pool_finalizado", [] { test_pool_finalizado(); });
    run("valvula_edad", [&] { test_valvula_edad(valves); });
    run("valvula_tope", [&] { test_valvula_tope(valves); });
    run("recientes", [&] { test_recientes(recent); });
    run("cajas_sin_subir", [] { test_cajas_sin_subir(); });
    run("render_target", [] { test_render_target(); });
#ifndef BASE
    run("huella_subida", [] { test_huella_subida(); });
#endif
    run("pool_de_texturas", [] { test_pool_de_texturas(); });
    CHECK(limbo_textures() == 0);
    printf("modo %s: %d comprobaciones, %d fallos\n", mode.c_str(), g_checked, g_failed);
    return g_failed ? 1 : 0;
}
