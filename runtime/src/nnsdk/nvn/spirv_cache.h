// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
// Caché de las traducciones Maxwell → SPIR-V, en memoria y en la SD (<datos>/spirv.bin).
//
// El juego crea varios programas con el mismo código (MH Rise: 1200 traducciones de 246 huellas por partida, 1800 de
// 348 tras un cambio de shaders) y cada una costaba 2,2 ms en NvnSubmit, con el frame parado: 2,6-4,2 s de tirones por
// partida, sobre todo al empezar y la primera vez que sale algo. Con la caché cada código distinto se traduce una vez
// por versión del traductor: las repetidas salen de la memoria y las de partidas anteriores, del fichero.
//
// La clave (128 bits) la calcula program.cpp con todo lo que decide la traducción. El fichero lleva la huella del
// traductor (Eden ya parcheado + el puente nvn_shader + sirit, ver runtime/recompiler/CMakeLists.txt): si no coincide
// se vacía. Se añade al final, desde otro hilo y sin fsync: un registro a medias (el proceso murió escribiendo) se
// descarta al leer, con todo lo que vaya detrás.
//
// Formato (little endian): cabecera de 32 bytes "NVSPIRV1", u32 formato, u32 0, u64 huella del traductor, u64 0;
// después registros: u32 'SPVR', u32 bytes, u64 clave.a, u64 clave.b, u64 FNV-1a de los bytes, y los bytes (serialize).

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nvn_shader.h"

namespace nvn::spirv_cache {

struct Key {
    uint64_t a = 0, b = 0;
    bool operator==(const Key& o) const { return a == o.a && b == o.b; }
};

struct KeyHash {
    size_t operator()(const Key& k) const { return static_cast<size_t>(k.a ^ (k.b * 0x9E3779B97F4A7C15ULL)); }
};

// Dos hashes de 64 bits distintos (FNV-1a y uno de multiplicar y rotar): 128 bits de clave.
class Hasher {
public:
    void bytes(const void* data, size_t size);
    // Solo enteros (sin relleno).
    void u32(uint32_t v) { bytes(&v, sizeof v); }
    void u64(uint64_t v) { bytes(&v, sizeof v); }
    Key key() const { return {a_, b_}; }

private:
    uint64_t a_ = 0xcbf29ce484222325ULL, b_ = 0x6a09e667f3bcc909ULL;
};

// Translated ⇄ bytes (sin el texto de error: solo se guardan traducciones buenas).
void serialize(const nvnshader::Translated& t, std::vector<uint8_t>& out);
bool deserialize(const uint8_t* data, size_t size, nvnshader::Translated& out);

class Cache {
public:
    Cache() = default;
    ~Cache();
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;

    // Empieza a leer el fichero en un hilo propio, que después escribe lo nuevo. Solo cuenta la primera llamada.
    void open(const std::string& path, uint64_t build);
    // La traducción con esa clave. Si el fichero aún se está leyendo, la primera vez espera como mucho 500 ms (una SD
    // atascada no para el frame más).
    bool find(const Key& key, nvnshader::Translated& out);
    // A la memoria y, si hay fichero, a la cola para añadirla (si no estaba ya en él).
    void insert(const Key& key, const nvnshader::Translated& t);
    // Espera a que el fichero esté leído (el precalentado de pipelines lo necesita entero). false si la caché no se abrió
    // o no terminó a tiempo.
    bool wait_loaded(std::chrono::milliseconds timeout);

    struct Stats {
        size_t loaded = 0;     // registros leídos del fichero
        size_t hits = 0;       // find que encontró
        size_t inserted = 0;   // insert nuevos
        size_t written = 0;    // registros añadidos al fichero
        uint64_t file_bytes = 0;
        bool rebuilt = false;  // el fichero no valía (otro traductor, otro formato o roto): se vació
        bool cut = false;      // tenía un final roto: se cortó
    };
    Stats stats();
    // Pruebas: espera a que se haya leído el fichero y escrito todo lo pendiente.
    void wait_idle();

private:
    void load();
    void write_loop();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<Key, std::vector<uint8_t>, KeyHash> entries_;  // traducciones serializadas
    std::unordered_set<Key, KeyHash> on_disk_;
    std::deque<std::pair<Key, std::vector<uint8_t>>> queue_;
    bool opened_ = false, loaded_ = false, gave_up_ = false, stop_ = false, writing_ = false;
    int fd_ = -1;
    uint64_t file_bytes_ = 0, build_ = 0;
    std::string path_;
    std::thread thread_;
    Stats stats_;
};

// La del proceso (no se destruye nunca).
Cache& global();

}  // namespace nvn::spirv_cache
