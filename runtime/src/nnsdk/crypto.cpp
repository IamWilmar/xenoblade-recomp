// nn::crypto: los hashes que importa el juego (SHA-1, SHA-256, SHA-512, MD5) y el generador aleatorio. Los
// *Impl son objetos del juego (los Sha256Generator… inline de la cabecera los tienen dentro), así que todo el estado
// vive en ellos con el layout del SDK: buffer, bytes pendientes, bits ya procesados y estado (1 iniciado, 2 con el
// hash calculado; GetHash vuelve a dar el mismo). Implementación propia (no hay OpenSSL en el NDK), comprobada con
// los vectores de FIPS 180 y RFC 1321.

#include <stdlib.h>

#include <cstdint>
#include <cstring>

#include "nnsdk/types.h"

namespace nn::crypto {

namespace {

uint32_t rotl32(uint32_t x, int n) { return x << n | x >> (32 - n); }
uint32_t rotr32(uint32_t x, int n) { return x >> n | x << (32 - n); }
uint64_t rotr64(uint64_t x, int n) { return x >> n | x << (64 - n); }

uint32_t load_be32(const uint8_t* p) { return uint32_t{p[0]} << 24 | uint32_t{p[1]} << 16 | uint32_t{p[2]} << 8 | p[3]; }
uint64_t load_be64(const uint8_t* p) { return uint64_t{load_be32(p)} << 32 | load_be32(p + 4); }
uint32_t load_le32(const uint8_t* p) { return uint32_t{p[3]} << 24 | uint32_t{p[2]} << 16 | uint32_t{p[1]} << 8 | p[0]; }

void store_be32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = static_cast<uint8_t>(v >> (24 - 8 * i));
}
void store_be64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = static_cast<uint8_t>(v >> (56 - 8 * i));
}

// ClearMemory del SDK: que el compilador no se salte el borrado de un objeto que muere.
void clear_memory(void* p, size_t size) {
    volatile uint8_t* b = static_cast<volatile uint8_t*>(p);
    while (size--) *b++ = 0;
}

void sha1_block(uint32_t h[5], const uint8_t* block) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) w[i] = load_be32(block + 4 * i);
    for (int i = 16; i < 80; i++) w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & c) | (~b & d);
            k = 0x5a827999;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ed9eba1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8f1bbcdc;
        } else {
            f = b ^ c ^ d;
            k = 0xca62c1d6;
        }
        const uint32_t t = rotl32(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = t;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
}

constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

void sha256_block(uint32_t h[8], const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = load_be32(block + 4 * i);
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t t1 = hh + (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) + ((e & f) ^ (~e & g)) + kSha256K[i] + w[i];
        const uint32_t t2 = (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

constexpr uint64_t kSha512K[80] = {
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538,
    0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe,
    0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2, 0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235,
    0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5, 0x983e5152ee66dfab,
    0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
    0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df, 0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218,
    0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8, 0x19a4c116b8d2d0c8, 0x1e376c085141ab53,
    0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b, 0xca273eceea26619c,
    0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6,
    0x113f9804bef90dae, 0x1b710b35131c471b, 0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
};

void sha512_block(uint64_t h[8], const uint8_t* block) {
    uint64_t w[80];
    for (int i = 0; i < 16; i++) w[i] = load_be64(block + 8 * i);
    for (int i = 16; i < 80; i++) {
        const uint64_t s0 = rotr64(w[i - 15], 1) ^ rotr64(w[i - 15], 8) ^ (w[i - 15] >> 7);
        const uint64_t s1 = rotr64(w[i - 2], 19) ^ rotr64(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint64_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 80; i++) {
        const uint64_t t1 = hh + (rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)) + ((e & f) ^ (~e & g)) + kSha512K[i] + w[i];
        const uint64_t t2 = (rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

void md5_block(uint32_t h[4], const uint8_t* block) {
    static const uint32_t kK[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    };
    static const int kShift[16] = {7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21};
    uint32_t m[16];
    for (int i = 0; i < 16; i++) m[i] = load_le32(block + 4 * i);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    for (int i = 0; i < 64; i++) {
        uint32_t f;
        int g;
        if (i < 16) {
            f = (b & c) | (~b & d);
            g = i;
        } else if (i < 32) {
            f = (d & b) | (~d & c);
            g = (5 * i + 1) & 15;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3 * i + 5) & 15;
        } else {
            f = c ^ (b | ~d);
            g = (7 * i) & 15;
        }
        const uint32_t t = d;
        d = c;
        c = b;
        b = b + rotl32(a + f + kK[i] + m[g], kShift[(i >> 4) * 4 + (i & 3)]);
        a = t;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
}

// Lo común de SHA-1/SHA-256: bloques de 64 bytes y la longitud en bits big-endian al final del último.
struct Sha1State {
    uint32_t h[5];
    uint8_t block[64];
    size_t buffered;
    uint64_t bits;
};

void sha1_update(Sha1State* s, const uint8_t* p, size_t size) {
    s->bits += ((s->buffered + size) << 3) & ~uint64_t{0x1ff};
    if (s->buffered) {
        const size_t n = size < 64 - s->buffered ? size : 64 - s->buffered;
        memcpy(s->block + s->buffered, p, n);
        s->buffered += n;
        p += n;
        size -= n;
        if (s->buffered != 64) return;
        sha1_block(s->h, s->block);
        s->buffered = 0;
    }
    for (; size >= 64; p += 64, size -= 64) sha1_block(s->h, p);
    if (size) {
        memcpy(s->block, p, size);
        s->buffered = size;
    }
}

}  // namespace

void GenerateSha1Hash(void* out, size_t, const void* data, size_t size) {
    Sha1State s{{0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0}, {}, 0, 0};
    sha1_update(&s, static_cast<const uint8_t*>(data), size);
    s.bits += s.buffered * 8;
    s.block[s.buffered++] = 0x80;
    if (s.buffered > 56) {
        memset(s.block + s.buffered, 0, 64 - s.buffered);
        sha1_block(s.h, s.block);
        memset(s.block, 0, 56);
    } else {
        memset(s.block + s.buffered, 0, 56 - s.buffered);
    }
    store_be64(s.block + 56, s.bits);
    sha1_block(s.h, s.block);
    // El SDK escribe los 20 bytes sin mirar el tamaño que le dan.
    for (int i = 0; i < 5; i++) store_be32(static_cast<uint8_t*>(out) + 4 * i, s.h[i]);
    clear_memory(&s, sizeof s);
}

void GenerateCryptographicallyRandomBytes(void* out, size_t size) { arc4random_buf(out, size); }

namespace detail {

class Sha256Impl {
public:
    ~Sha256Impl();
    void Initialize();
    void Update(const void* data, size_t size);
    void GetHash(void* out, size_t size);

private:
    uint32_t h_[8];
    uint8_t block_[64];
    size_t buffered_;
    uint64_t bits_;  // de los bloques ya procesados (Update los suma por adelantado)
    int state_;
};
static_assert(sizeof(Sha256Impl) == 0x78);

class Sha512Impl {
public:
    ~Sha512Impl();
    void Initialize();
    void Update(const void* data, size_t size);
    void GetHash(void* out, size_t size);

private:
    uint64_t h_[8];
    uint8_t block_[128];
    size_t buffered_;
    uint64_t bits_high_, bits_low_;  // longitud de 128 bits; Update solo suma a la baja, como el SDK
    int state_;
};
static_assert(sizeof(Sha512Impl) == 0xe0);

class Md5Impl {
public:
    ~Md5Impl();
    void Initialize();
    void Update(const void* data, size_t size);
    void GetHash(void* out, size_t size);

private:
    uint32_t h_[4];
    uint8_t block_[64];
    uint64_t total_;  // bytes recibidos (el buffer va por total_ % 64)
    int state_;
};
static_assert(sizeof(Md5Impl) == 0x60);

Sha256Impl::~Sha256Impl() { clear_memory(this, sizeof *this); }

void Sha256Impl::Initialize() {
    static const uint32_t kInit[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(h_, kInit, sizeof h_);
    buffered_ = 0;
    bits_ = 0;
    state_ = 1;
}

void Sha256Impl::Update(const void* data, size_t size) {
    auto p = static_cast<const uint8_t*>(data);
    bits_ += ((buffered_ + size) << 3) & ~uint64_t{0x1ff};
    if (buffered_) {
        const size_t n = size < 64 - buffered_ ? size : 64 - buffered_;
        memcpy(block_ + buffered_, p, n);
        buffered_ += n;
        p += n;
        size -= n;
        if (buffered_ != 64) return;
        sha256_block(h_, block_);
        buffered_ = 0;
    }
    for (; size >= 64; p += 64, size -= 64) sha256_block(h_, p);
    if (size) {
        memcpy(block_, p, size);
        buffered_ = size;
    }
}

void Sha256Impl::GetHash(void* out, size_t) {
    if (state_ == 1) {
        bits_ += buffered_ * 8;
        block_[buffered_++] = 0x80;
        if (buffered_ > 56) {
            memset(block_ + buffered_, 0, 64 - buffered_);
            sha256_block(h_, block_);
            memset(block_, 0, 56);
        } else {
            memset(block_ + buffered_, 0, 56 - buffered_);
        }
        store_be64(block_ + 56, bits_);
        sha256_block(h_, block_);
        state_ = 2;
    }
    for (int i = 0; i < 8; i++) store_be32(static_cast<uint8_t*>(out) + 4 * i, h_[i]);
}

Sha512Impl::~Sha512Impl() { clear_memory(this, sizeof *this); }

void Sha512Impl::Initialize() {
    static const uint64_t kInit[8] = {0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
                                      0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
                                      0x1f83d9abfb41bd6b, 0x5be0cd19137e2179};
    memcpy(h_, kInit, sizeof h_);
    buffered_ = 0;
    bits_high_ = bits_low_ = 0;
    state_ = 1;
}

void Sha512Impl::Update(const void* data, size_t size) {
    auto p = static_cast<const uint8_t*>(data);
    bits_low_ += ((buffered_ + size) << 3) & ~uint64_t{0x3ff};
    if (buffered_) {
        const size_t n = size < 128 - buffered_ ? size : 128 - buffered_;
        memcpy(block_ + buffered_, p, n);
        buffered_ += n;
        p += n;
        size -= n;
        if (buffered_ != 128) return;
        sha512_block(h_, block_);
        buffered_ = 0;
    }
    for (; size >= 128; p += 128, size -= 128) sha512_block(h_, p);
    if (size) {
        memcpy(block_, p, size);
        buffered_ = size;
    }
}

void Sha512Impl::GetHash(void* out, size_t) {
    if (state_ == 1) {
        bits_low_ += buffered_ * 8;
        block_[buffered_++] = 0x80;
        if (buffered_ > 112) {
            memset(block_ + buffered_, 0, 128 - buffered_);
            sha512_block(h_, block_);
            memset(block_, 0, 112);
        } else {
            memset(block_ + buffered_, 0, 112 - buffered_);
        }
        store_be64(block_ + 112, bits_high_);
        store_be64(block_ + 120, bits_low_);
        sha512_block(h_, block_);
        state_ = 2;
    }
    for (int i = 0; i < 8; i++) store_be64(static_cast<uint8_t*>(out) + 8 * i, h_[i]);
}

Md5Impl::~Md5Impl() { clear_memory(this, sizeof *this); }

void Md5Impl::Initialize() {
    h_[0] = 0x67452301;
    h_[1] = 0xefcdab89;
    h_[2] = 0x98badcfe;
    h_[3] = 0x10325476;
    total_ = 0;
    state_ = 1;
}

void Md5Impl::Update(const void* data, size_t size) {
    auto p = static_cast<const uint8_t*>(data);
    const size_t used = total_ & 63, room = 64 - used;
    total_ += size;
    if (room > size) {
        memcpy(block_ + used, p, size);
        return;
    }
    memcpy(block_ + used, p, room);
    md5_block(h_, block_);
    p += room;
    size -= room;
    for (; size >= 64; p += 64, size -= 64) md5_block(h_, p);
    memcpy(block_, p, size);
}

void Md5Impl::GetHash(void* out, size_t) {
    if (state_ == 1) {
        const uint64_t total = total_;
        // El 0x80 cuenta como un byte más en total_ (así lo lleva el SDK).
        if ((total_++ & 63) == 63) {
            block_[63] = 0x80;
            md5_block(h_, block_);
        } else {
            block_[total & 63] = 0x80;
        }
        size_t used = total_ & 63;
        if (used > 56) {
            memset(block_ + used, 0, 64 - used);
            md5_block(h_, block_);
            used = 0;
        }
        memset(block_ + used, 0, 56 - used);
        const uint32_t bits_low = static_cast<uint32_t>(total << 3), bits_high = static_cast<uint32_t>(total >> 29);
        memcpy(block_ + 56, &bits_low, 4);  // little-endian
        memcpy(block_ + 60, &bits_high, 4);
        md5_block(h_, block_);
        state_ = 2;
    }
    memcpy(out, h_, 16);
}

}  // namespace detail

}  // namespace nn::crypto
