// nn::hid: el grafo de vibración del SDK (VibrationNode y sus derivados). El juego crea los nodos en su propia memoria,
// muchos dentro de objetos suyos (un VibrationPlayer en +0x40 y su siguiente campo en +0x110, main+0x6f8994c), y los
// conecta con VibrationNodeConnection: reproductores de .bnvib y escritores de valores sueltos que pasan por
// mezcladores hasta un VibrationTarget por actuador. Un hilo suyo llama a VibrationNode::Update() cada
// DefaultVibrationSampleInterval (5 ms, main+0x6f8ba28), y desde el juego se lee de vuelta GetCurrentVibration,
// IsPlaying y GetCurrentPosition para saber cuánto queda de cada vibración. Por eso el grafo se calcula entero como en
// el SDK; lo único que no existe es el motor: VibrationTarget no manda el valor a ningún mando.
//
// Las clases tienen la disposición del SDK (0x40, 0x48, 0x50, 0xd0 y 0x38 bytes, vtable con los destructores y
// OnNextSampleRequired) porque el juego las reserva con esos tamaños. Un solo mutex recursivo guarda todo el grafo: el
// SDK usa varios, pero aquí nada llama al juego mientras se tiene.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>

#include "nnsdk/runtime.h"
#include "nnsdk/types.h"

#define HIDDEN __attribute__((visibility("hidden")))

namespace nn::hid {

struct VibrationValue {
    float amplitude_low, frequency_low, amplitude_high, frequency_high;
};
struct VibrationModulation {
    float gain_low, pitch_low, gain_high, pitch_high;
};
struct VibrationDeviceHandle {
    uint32_t storage;
};
enum VibrationMixMode : int {
    VibrationMixMode_None = 0,
    VibrationMixMode_MaxAmplitude = 1,            // la entrada con más amplitud total, entera
    VibrationMixMode_MaxAmplitudePerSubband = 2,  // la mayor de cada banda por separado
    VibrationMixMode_AmplitudeSum = 3,            // suma de amplitudes, frecuencia media ponderada
};

// Cabecera de un .bnvib tal como la devuelve GetFileInfo (el juego lee sampleLength y samplingRate, main+0x6f89a34).
struct VibrationFileInfo {
    uint32_t meta_data_size;
    uint16_t format_id;
    uint16_t sampling_rate;
    uint32_t data_size;
    int32_t sample_length;
    int32_t is_loop;
    int32_t loop_start_position;
    int32_t loop_end_position;
    int32_t loop_interval;
};
static_assert(sizeof(VibrationFileInfo) == 0x20);

namespace detail {
// Nodo de nn::util::IntrusiveList: lista circular con cabecera, {anterior, siguiente}.
struct ListNode {
    ListNode* prev;
    ListNode* next;
};
}  // namespace detail

class VibrationNodeConnection;

// Base: enlace en la lista global de nodos (+0x08), valor actual (+0x18), lista de conexiones que llegan (+0x28) y la
// muestra en la que se calculó (+0x38). Los derivados empiezan en +0x39, dentro del relleno del final de la base.
class VibrationNode : public detail::ListNode {
public:
    static void Update();
    VibrationValue GetCurrentVibration() const;
    // Por si el juego borra uno con delete a través de la vtable: la memoria es de su allocator.
    HIDDEN static void operator delete(void* p) { nnsdk::host().game_delete(p); }

protected:
    HIDDEN VibrationNode();
    HIDDEN virtual ~VibrationNode();
    HIDDEN virtual void OnNextSampleRequired(VibrationValue* value, detail::ListNode* inputs);
    HIDDEN void SetCurrentVibration(const VibrationValue& value);

private:
    friend class VibrationNodeConnection;
    friend class VibrationMixer;
    HIDDEN void SyncSampleNumber();

    VibrationValue value_;
    detail::ListNode inputs_;
    uint8_t sample_;
};

class VibrationNodeConnection : public detail::ListNode {
public:
    VibrationNodeConnection();
    virtual ~VibrationNodeConnection();
    bool Connect(VibrationNode* source, VibrationNode* destination);
    void Disconnect();
    bool IsConnected() const;
    void SetModulation(const VibrationModulation& modulation);
    HIDDEN static void operator delete(void* p) { nnsdk::host().game_delete(p); }

private:
    friend class VibrationNode;
    friend class VibrationMixer;
    VibrationNode* source_;
    VibrationNode* destination_;
    VibrationModulation modulation_;
};

class VibrationMixer : public VibrationNode {
public:
    VibrationMixer();
    ~VibrationMixer() override;
    void SetMixMode(const VibrationMixMode& mode);
    void SetMixMode(VibrationMixMode mode);  // SDK 9 (Xenoblade): por valor

protected:
    HIDDEN void OnNextSampleRequired(VibrationValue* value, detail::ListNode* inputs) override;

private:
    // Compresor de AmplitudeSum: por encima del umbral, la amplitud se divide entre la razón. Nada lo cambia (umbral 0 y
    // razón 1: no comprime); son constantes para que el mixer mida 0x40 como en el SDK 9 de Xenoblade, que pone sus
    // VibrationTarget cada 0x48 bytes (con dos campos más aquí, SetVibrationDeviceHandle de uno pisaba el vptr del
    // siguiente y VibrationNode::Update saltaba a la basura).
    static constexpr float compressor_threshold_ = 0.0f;
    static constexpr float compressor_ratio_ = 1.0f;
    int32_t mix_mode_;
};

class VibrationTarget : public VibrationMixer {
public:
    VibrationTarget();
    ~VibrationTarget() override;
    void SetVibrationDeviceHandle(const VibrationDeviceHandle& handle);
    void UnsetVibrationDeviceHandle();
    bool IsVibrationDeviceHandleSet() const;

protected:
    HIDDEN void OnNextSampleRequired(VibrationValue* value, detail::ListNode* inputs) override;

private:
    bool handle_set_;
    VibrationDeviceHandle handle_;
};

class VibrationWriter : public VibrationNode {
public:
    VibrationWriter();
    ~VibrationWriter() override;
    void Write(const VibrationValue& value);

protected:
    HIDDEN void OnNextSampleRequired(VibrationValue* value, detail::ListNode* inputs) override;
};

class VibrationPlayer : public VibrationNode {
public:
    VibrationPlayer();
    ~VibrationPlayer() override;
    Result SetVibrationFileAddress(const void* address, size_t size);
    void UnsetVibrationFileAddress();
    bool IsPlayable() const;
    VibrationFileInfo GetFileInfo() const;
    void Play();
    void Stop();
    bool IsPlaying() const;
    void SetCurrentPosition(int position);
    int GetCurrentPosition() const;
    void SetLoop(bool loop);
    void SetLoopStartPosition(int position);
    void SetLoopEndPosition(int position);
    void SetLoopInterval(int interval);
    // Nombres del SDK 9 (Xenoblade) para SetVibrationFileAddress e IsPlayable.
    Result Load(const void* address, size_t size);
    bool IsLoaded() const;
    // El SDK 9 declara OnNextSampleRequired con IntrusiveList<VibrationNodeConnection>* (la misma cabecera de lista):
    // el juego lo importa con ese nombre (lo llama desde sus derivados).
    HIDDEN static void next_sample_sdk9(VibrationPlayer* p, VibrationValue* value, detail::ListNode* inputs) {
        p->VibrationPlayer::OnNextSampleRequired(value, inputs);
    }

protected:
    HIDDEN void OnNextSampleRequired(VibrationValue* value, detail::ListNode* inputs) override;

private:
    HIDDEN void Retrieve(VibrationValue* value, int index) const;

    bool playable_;
    bool playing_;
    bool loop_;
    int32_t position_;     // muestra que suena ahora
    int32_t next_;         // la siguiente; negativa durante la pausa entre vueltas del bucle
    int32_t loop_start_;
    int32_t loop_end_;
    int32_t loop_interval_;
    int32_t delay_;        // muestras de silencio al empezar (SetDelayTime, que MH Rise no usa)
    int32_t delay_left_;
    float speed_;
    float speed_acc_;      // muestras por consumir: avanza speed_ en cada Update
    VibrationFileInfo info_;
    // VibrationFileParserContext: las muestras y cuántas hay; el resto del SDK no se usa.
    const uint8_t* data_;
    [[maybe_unused]] uint32_t reserved_;
    int32_t sample_count_;
    [[maybe_unused]] uint8_t parser_reserved_[0x30];  // 0x30 y no 0x40: el player del SDK 9 mide 0xc0 (el juego deriva de él)
};

static_assert(sizeof(VibrationNode) == 0x40);
static_assert(sizeof(VibrationNodeConnection) == 0x38);
// Tamaños del SDK 9.3 (Xenoblade): targets cada 0x48 (main+0x698340, main+0x6b03a0) y su derivado del player escribe
// sus campos desde +0xc0 (main+0x6982b4).
static_assert(sizeof(VibrationMixer) == 0x40);
static_assert(sizeof(VibrationTarget) == 0x48);
static_assert(sizeof(VibrationWriter) == 0x40);
static_assert(sizeof(VibrationPlayer) == 0xc0);

}  // namespace nn::hid

using namespace nn;
using namespace nn::hid;

namespace {

// Lo que vale un nodo sin vibración: amplitud 0 en las dos bandas, a 160 y 320 Hz.
constexpr VibrationValue kSilence{0.0f, 160.0f, 0.0f, 320.0f};

// Errores de ParseVibrationFile (módulo hid, 202).
constexpr Result kResultInvalidVibrationFile{202 | 141 << 9};
constexpr Result kResultUnsupportedFormat{202 | 142 << 9};
constexpr Result kResultUnsupportedSamplingRate{202 | 143 << 9};
constexpr Result kResultTruncatedData{202 | 144 << 9};
constexpr Result kResultInvalidLoopRange{202 | 145 << 9};

std::recursive_mutex& graph_mutex() {
    static std::recursive_mutex m;
    return m;
}

detail::ListNode& all_nodes() {
    static detail::ListNode head{&head, &head};
    return head;
}

// Número de la muestra en curso (VibrationNode::Update lo avanza): un nodo ya calculado en esta muestra no se repite
// aunque lo alcancen varias conexiones.
uint8_t g_sample = 0;

void link_back(detail::ListNode& head, detail::ListNode& node) {
    node.prev = head.prev;
    node.next = &head;
    head.prev->next = &node;
    head.prev = &node;
}

void unlink(detail::ListNode& node) {
    node.prev->next = node.next;
    node.next->prev = node.prev;
    node.prev = node.next = &node;
}

uint32_t read32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

// El formato .bnvib igual que nn::hid::detail::ParseVibrationFile: tamaño de la metainformación (4, 12 o 16),
// formato 3, 200 muestras por segundo, inicio y fin del bucle y pausa entre vueltas si los hay, y luego el tamaño de
// los datos y las muestras de 4 bytes.
Result parse_vibration_file(VibrationFileInfo* info, const uint8_t** data, int32_t* count, const uint8_t* file,
                            size_t size) {
    if (!file || size < 8) return kResultInvalidVibrationFile;
    info->meta_data_size = read32(file);
    memcpy(&info->format_id, file + 4, 2);
    memcpy(&info->sampling_rate, file + 6, 2);
    if (info->meta_data_size < 4) return kResultInvalidVibrationFile;
    if (info->format_id != 3) return kResultUnsupportedFormat;
    if (info->sampling_rate != 200) return kResultUnsupportedSamplingRate;
    const bool loop = info->meta_data_size >= 0xc;
    info->is_loop = loop;
    if (loop) {
        if (size < 0x10) return kResultInvalidVibrationFile;
        info->loop_start_position = static_cast<int32_t>(read32(file + 8));
        info->loop_end_position = static_cast<int32_t>(read32(file + 0xc));
        info->loop_interval = 0;
        if (read32(file + 8) > read32(file + 0xc)) return kResultInvalidLoopRange;
        if (info->meta_data_size >= 0x10) {
            if (size < 0x14) return kResultInvalidVibrationFile;
            info->loop_interval = static_cast<int32_t>(read32(file + 0x10));
            if (info->loop_interval < 0) return kResultInvalidLoopRange;
        }
    }
    const size_t header = size_t{info->meta_data_size} + 8;
    if (header > size) return kResultInvalidVibrationFile;
    info->data_size = read32(file + info->meta_data_size + 4);
    info->sample_length = static_cast<int32_t>(info->data_size >> 2);
    if (loop) {
        if (static_cast<uint32_t>(info->loop_end_position) > static_cast<uint32_t>(info->sample_length))
            return kResultInvalidLoopRange;
    } else {
        info->loop_start_position = 0;
        info->loop_end_position = info->sample_length;
        info->loop_interval = 0;
    }
    *data = file + header;
    *count = info->sample_length;
    return header + info->data_size > size ? kResultTruncatedData : ResultSuccess;
}

}  // namespace

namespace nn::hid {

// --- VibrationNode ---

VibrationNode::VibrationNode() : detail::ListNode{this, this}, value_(kSilence), inputs_{&inputs_, &inputs_} {
    std::lock_guard lock(graph_mutex());
    sample_ = g_sample;
    link_back(all_nodes(), *this);
}

// Como el SDK, solo sale de la lista global: las conexiones que lo usen son cosa del juego (las desconecta antes).
VibrationNode::~VibrationNode() {
    std::lock_guard lock(graph_mutex());
    if (next != this) unlink(*this);
}

void VibrationNode::OnNextSampleRequired(VibrationValue*, detail::ListNode*) {}

VibrationValue VibrationNode::GetCurrentVibration() const {
    std::lock_guard lock(graph_mutex());
    return value_;
}

void VibrationNode::SetCurrentVibration(const VibrationValue& value) {
    std::lock_guard lock(graph_mutex());
    value_ = value;
}

// Primero las fuentes (recursivo), luego este nodo si aún no tiene la muestra actual.
void VibrationNode::SyncSampleNumber() {
    for (detail::ListNode* n = inputs_.next; n != &inputs_; n = n->next) {
        if (VibrationNode* source = static_cast<VibrationNodeConnection*>(n)->source_) source->SyncSampleNumber();
    }
    if (sample_ == g_sample) return;
    VibrationValue value = value_;
    OnNextSampleRequired(&value, &inputs_);
    value_ = value;
    sample_ = g_sample;
}

void VibrationNode::Update() {
    std::lock_guard lock(graph_mutex());
    ++g_sample;
    detail::ListNode& head = all_nodes();
    for (detail::ListNode* n = head.next; n != &head; n = n->next) static_cast<VibrationNode*>(n)->SyncSampleNumber();
    // Aquí el SDK manda a los mandos lo que dejaron los VibrationTarget: la Flip no tiene dónde.
}

// --- VibrationNodeConnection ---

VibrationNodeConnection::VibrationNodeConnection()
    : detail::ListNode{this, this}, source_(nullptr), destination_(nullptr), modulation_{1.0f, 1.0f, 1.0f, 1.0f} {}

VibrationNodeConnection::~VibrationNodeConnection() { Disconnect(); }

// Falla si ya está conectada o si el destino ya tiene una conexión desde esa fuente.
bool VibrationNodeConnection::Connect(VibrationNode* source, VibrationNode* destination) {
    std::lock_guard lock(graph_mutex());
    if (destination_ || !destination) return false;
    for (detail::ListNode* n = destination->inputs_.next; n != &destination->inputs_; n = n->next) {
        if (static_cast<VibrationNodeConnection*>(n)->source_ == source) return false;
    }
    source_ = source;
    destination_ = destination;
    link_back(destination->inputs_, *this);
    return true;
}

void VibrationNodeConnection::Disconnect() {
    std::lock_guard lock(graph_mutex());
    if (!destination_) return;
    if (next != this) unlink(*this);
    source_ = destination_ = nullptr;
}

bool VibrationNodeConnection::IsConnected() const {
    std::lock_guard lock(graph_mutex());
    return destination_ != nullptr;
}

void VibrationNodeConnection::SetModulation(const VibrationModulation& modulation) {
    std::lock_guard lock(graph_mutex());
    modulation_ = modulation;
}

// --- VibrationMixer ---

VibrationMixer::VibrationMixer()
    : mix_mode_(VibrationMixMode_MaxAmplitudePerSubband) {}
VibrationMixer::~VibrationMixer() = default;

void VibrationMixer::SetMixMode(const VibrationMixMode& mode) {
    std::lock_guard lock(graph_mutex());
    mix_mode_ = mode;
}
void VibrationMixer::SetMixMode(VibrationMixMode mode) {
    std::lock_guard lock(graph_mutex());
    mix_mode_ = mode;
}

// Cada entrada es el valor actual de su fuente con la modulación de la conexión (ganancia en la amplitud, tono en la
// frecuencia); luego el compresor, que nunca deja pasar de la razón máxima del SDK (60: por encima, se corta en seco).
void VibrationMixer::OnNextSampleRequired(VibrationValue* value, detail::ListNode* inputs) {
    std::lock_guard lock(graph_mutex());
    *value = kSilence;
    float weighted_low = 0.0f, weighted_high = 0.0f;
    for (detail::ListNode* n = inputs->next; n != inputs; n = n->next) {
        auto* connection = static_cast<VibrationNodeConnection*>(n);
        if (mix_mode_ == VibrationMixMode_None || !connection->source_) continue;
        const VibrationValue in = connection->source_->value_;
        const VibrationModulation& m = connection->modulation_;
        const float low = in.amplitude_low * m.gain_low, high = in.amplitude_high * m.gain_high;
        const float freq_low = in.frequency_low * m.pitch_low, freq_high = in.frequency_high * m.pitch_high;
        switch (mix_mode_) {
            case VibrationMixMode_MaxAmplitude:
                if (low + high >= value->amplitude_low + value->amplitude_high)
                    *value = {low, freq_low, high, freq_high};
                break;
            case VibrationMixMode_MaxAmplitudePerSubband:
                if (low >= value->amplitude_low) value->amplitude_low = low, value->frequency_low = freq_low;
                if (high >= value->amplitude_high) value->amplitude_high = high, value->frequency_high = freq_high;
                break;
            case VibrationMixMode_AmplitudeSum:
                value->amplitude_low += low;
                value->amplitude_high += high;
                weighted_low += low * freq_low;
                weighted_high += high * freq_high;
                break;
        }
    }
    if (mix_mode_ == VibrationMixMode_AmplitudeSum) {
        if (value->amplitude_low > 0.0f) value->frequency_low = weighted_low / value->amplitude_low;
        if (value->amplitude_high > 0.0f) value->frequency_high = weighted_high / value->amplitude_high;
    }
    auto compress = [this](float amplitude) {
        if (amplitude < compressor_threshold_) return amplitude;
        if (compressor_ratio_ >= 60.0f) return compressor_threshold_;
        return compressor_threshold_ + (amplitude - compressor_threshold_) / compressor_ratio_;
    };
    value->amplitude_low = compress(value->amplitude_low);
    value->amplitude_high = compress(value->amplitude_high);
}

// --- VibrationTarget: el mezclador final de un actuador ---

VibrationTarget::VibrationTarget() : handle_set_(false), handle_{0} {}
VibrationTarget::~VibrationTarget() = default;

void VibrationTarget::SetVibrationDeviceHandle(const VibrationDeviceHandle& handle) {
    std::lock_guard lock(graph_mutex());
    handle_set_ = true;
    handle_ = handle;
}

// El SDK manda antes un valor de silencio al actuador para que no se quede vibrando; aquí no hay actuador.
void VibrationTarget::UnsetVibrationDeviceHandle() {
    std::lock_guard lock(graph_mutex());
    handle_set_ = false;
    handle_ = {0};
}

bool VibrationTarget::IsVibrationDeviceHandleSet() const {
    std::lock_guard lock(graph_mutex());
    return handle_set_;
}

// Sin handle conserva el último valor, como el SDK. Con handle mezcla; el SDK además lo encola para el mando.
void VibrationTarget::OnNextSampleRequired(VibrationValue* value, detail::ListNode* inputs) {
    std::lock_guard lock(graph_mutex());
    if (handle_set_) VibrationMixer::OnNextSampleRequired(value, inputs);
}

// --- VibrationWriter: el valor que escribe el juego, tal cual ---

VibrationWriter::VibrationWriter() = default;
VibrationWriter::~VibrationWriter() = default;
void VibrationWriter::Write(const VibrationValue& value) { SetCurrentVibration(value); }
void VibrationWriter::OnNextSampleRequired(VibrationValue*, detail::ListNode*) {}

// --- VibrationPlayer: reproduce un .bnvib a una muestra por Update ---

VibrationPlayer::VibrationPlayer()
    : playable_(false), playing_(false), loop_(false), position_(0), next_(0), loop_start_(0), loop_end_(0),
      loop_interval_(0), delay_(0), delay_left_(0), speed_(1.0f), speed_acc_(0.0f), info_{}, data_(nullptr),
      reserved_(0), sample_count_(0), parser_reserved_{} {}
VibrationPlayer::~VibrationPlayer() = default;

Result VibrationPlayer::SetVibrationFileAddress(const void* address, size_t size) {
    std::lock_guard lock(graph_mutex());
    playable_ = false;
    Result r = parse_vibration_file(&info_, &data_, &sample_count_, static_cast<const uint8_t*>(address), size);
    if (r.value) return r;
    playable_ = true;
    playing_ = false;
    position_ = next_ = 0;
    loop_ = info_.is_loop != 0;
    SetCurrentVibration(kSilence);
    loop_start_ = info_.loop_start_position;
    loop_end_ = info_.loop_end_position;
    loop_interval_ = info_.loop_interval;
    return ResultSuccess;
}

void VibrationPlayer::UnsetVibrationFileAddress() {
    std::lock_guard lock(graph_mutex());
    playable_ = false;
}

bool VibrationPlayer::IsPlayable() const { return playable_; }
Result VibrationPlayer::Load(const void* address, size_t size) { return SetVibrationFileAddress(address, size); }
bool VibrationPlayer::IsLoaded() const { return IsPlayable(); }
bool VibrationPlayer::IsPlaying() const { return playing_; }
int VibrationPlayer::GetCurrentPosition() const { return position_; }

VibrationFileInfo VibrationPlayer::GetFileInfo() const {
    std::lock_guard lock(graph_mutex());
    return info_;
}

void VibrationPlayer::Play() {
    std::lock_guard lock(graph_mutex());
    if (!playable_) return;
    playing_ = true;
    delay_left_ = delay_;
    speed_acc_ = 1.0f;
}

// Se queda en la muestra siguiente: Play() sigue desde ahí (main+0x6f89c84 guarda la posición y la restaura).
void VibrationPlayer::Stop() {
    std::lock_guard lock(graph_mutex());
    if (!playable_) return;
    position_ = next_;
    playing_ = false;
    SetCurrentVibration(kSilence);
}

void VibrationPlayer::SetCurrentPosition(int position) {
    if (position < 0) return;  // el SDK aborta
    std::lock_guard lock(graph_mutex());
    if (!playable_) return;
    position_ = next_ = position;
    if (!playing_) return;
    VibrationValue value;
    Retrieve(&value, position);
    SetCurrentVibration(value);
}

void VibrationPlayer::SetLoop(bool loop) {
    std::lock_guard lock(graph_mutex());
    if (playable_) loop_ = loop;
}

void VibrationPlayer::SetLoopStartPosition(int position) {
    std::lock_guard lock(graph_mutex());
    if (playable_ && position >= 0) loop_start_ = position;
}

void VibrationPlayer::SetLoopEndPosition(int position) {
    std::lock_guard lock(graph_mutex());
    if (playable_ && position >= 0) loop_end_ = position;
}

void VibrationPlayer::SetLoopInterval(int interval) {
    std::lock_guard lock(graph_mutex());
    if (playable_ && interval >= 0) loop_interval_ = interval;
}

// Muestra de 4 bytes: amplitud baja, frecuencia baja, amplitud alta, frecuencia alta. Amplitud en 1/255; frecuencia
// con 3 bits de octava (10 Hz << n) y 5 de semitono fino (2^(k/32)).
void VibrationPlayer::Retrieve(VibrationValue* value, int index) const {
    if (!data_ || index < 0 || index >= sample_count_) {
        *value = kSilence;
        return;
    }
    const uint8_t* s = data_ + size_t(index) * 4;
    auto frequency = [](uint8_t b) { return std::exp2(float(b & 0x1F) / 32.0f) * float(10 << (b >> 5)); };
    *value = {s[0] / 255.0f, frequency(s[1]), s[2] / 255.0f, frequency(s[3])};
}

// La misma máquina que el SDK: la espera inicial, luego tantas muestras como diga la velocidad acumulada; con bucle,
// al llegar al fin salta al inicio o, si hay pausa, a -pausa para contar muestras de silencio hasta volver a 0; sin
// bucle, al acabar para y vuelve a la posición 0.
void VibrationPlayer::OnNextSampleRequired(VibrationValue* value, detail::ListNode*) {
    std::lock_guard lock(graph_mutex());
    if (!playable_) {
        *value = kSilence;
        return;
    }
    if (delay_left_ > 0) {
        *value = kSilence;
        delay_left_--;
        return;
    }
    float acc = speed_acc_;
    while (acc >= 1.0f) {
        if (playing_) {
            int next = next_;
            if (loop_) {
                if (next < -loop_interval_ || next >= loop_end_)
                    next = next_ = loop_interval_ >= 1 ? -loop_interval_ : loop_start_;
            } else if (next >= info_.sample_length) {
                position_ = next_ = 0;
                playing_ = false;
                *value = kSilence;
                return;
            }
            if (position_ < 0 && next == 0) next = next_ = loop_start_;
            Retrieve(value, next);
            position_ = next_;
            next_ = position_ + 1;
        }
        acc -= 1.0f;
        speed_acc_ = acc;
    }
    speed_acc_ = acc + speed_;
}

}  // namespace nn::hid

// VibrationPlayer::OnNextSampleRequired(VibrationValue*, IntrusiveList<VibrationNodeConnection, …>*) del SDK 9.
extern "C" void
_ZN2nn3hid15VibrationPlayer20OnNextSampleRequiredEPNS0_14VibrationValueEPNS_4util13IntrusiveListINS0_23VibrationNodeConnectionENS4_27IntrusiveListBaseNodeTraitsIS6_S6_EEEE(
    nn::hid::VibrationPlayer* player, nn::hid::VibrationValue* value, nn::hid::detail::ListNode* inputs) {
    nn::hid::VibrationPlayer::next_sample_sdk9(player, value, inputs);
}
