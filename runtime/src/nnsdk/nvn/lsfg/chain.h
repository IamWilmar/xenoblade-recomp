// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 lsfg-vk
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once
// La cadena de compute de LSFG (modo rendimiento), port de Eden (src/video_core/renderer_vulkan/present/
// lsfg_{common,mipmaps,alpha,beta,gamma,delta,generate,chain}) a Vulkan a pelo sobre nuestro VkContext. Solo
// 2x: un frame generado (timestamp 0.5) entre cada par de frames reales, escrito en una imagen nuestra del
// tamaño de la ventana del juego. Mismos tamaños de grupo, orden de dispatches y orden de descriptores que Eden.
// Sin nullDescriptor: las entradas que faltan (la primera iteración) apuntan a una imagen 1x1 a cero.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "nnsdk/nvn/lsfg/dll.h"
#include "nnsdk/nvn/vk.h"

namespace nvn::lsfg {

constexpr int kMipLevels = 7, kHistory = 3, kAlphaStages = 4, kBetaStages = 5, kBetaOutputs = 6;
constexpr int kGammaStages = 5, kDeltaStages = 10, kDeltaInstances = 3, kFirstDeltaLevel = 4;

// Una pasada de compute: un binding por descriptor, en el orden de `types`.
struct Pass {
    uint32_t id = 0;
    std::vector<VkDescriptorType> types;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

// Los 25 pipelines del modo rendimiento, uno por shader. Solo hacen falta vkCreate*: se crean en un hilo aparte.
class Pipelines {
public:
    // false (y nada creado) si un shader no pasa la auditoría, sus bindings no son los que espera la cadena
    // o el driver no crea el módulo o el pipeline.
    bool create(const Modules& modules, bool float16, std::string* why);
    void destroy();
    const Pass& get(uint32_t id) const { return passes_.at(id); }
    size_t size() const { return passes_.size(); }

private:
    std::map<uint32_t, Pass> passes_;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED, view_format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageCreateFlags flags = 0;
    VkDeviceSize offset = 0;
};

class Chain {
public:
    Chain() = default;
    Chain(const Chain&) = delete;
    Chain& operator=(const Chain&) = delete;
    ~Chain() { destroy(); }

    // Para una ventana de ese tamaño y formato y una escala del flujo óptico (0.5 o 1.0). Todo en una reserva de
    // memoria. false con el motivo (y nada creado) si falla.
    bool create(const Pipelines& pipelines, VkExtent2D extent, VkFormat format, float flow, std::string* why);
    void destroy();

    // La primera vez que se graba: todas las imágenes a GENERAL y la imagen vacía a cero.
    bool initialized() const { return initialized_; }
    void record_init(VkCommandBuffer cmd);
    // Frame real n: su copia va en input(n). Las pasadas compartidas (mipmaps, alpha, beta) piden la copia de n;
    // la generación, además, las compartidas de n, n-1 y n-2. Deja G(n-1, n) en output().
    const Image& input(uint64_t n) const { return frames_[n % 2]; }
    const Image& output() const { return output_; }
    void record_shared(VkCommandBuffer cmd, uint64_t n) const;
    void record_generate(VkCommandBuffer cmd, uint64_t n) const;

    VkExtent2D extent() const { return extent_; }
    VkFormat format() const { return format_; }
    float flow() const { return flow_; }
    size_t images() const { return all_.size(); }
    VkDeviceSize bytes() const { return bytes_; }
    uint32_t sets() const { return set_count_; }

private:
    struct Alpha {
        Image temp1, temp2, temp3[2], out[kHistory][2];
        VkDescriptorSet sets[kAlphaStages - 1]{}, last[kHistory]{};
    };
    struct Beta {
        Image temp1[2], temp2[2], out[kBetaOutputs];
        VkDescriptorSet first[kHistory]{}, sets[kBetaStages - 1]{};
    };
    struct Gamma {
        Image temp1[3], temp2[2], out;
        VkDescriptorSet first[kHistory]{}, sets[kGammaStages - 1]{};
    };
    struct Delta {
        Image temp1[3], temp2[2], out1, out2;
        VkDescriptorSet first[kHistory]{}, sixth[kHistory]{}, sets[kDeltaStages - 2]{};
    };

    void add(Image& image, VkExtent2D extent, VkFormat format, VkFormat view_format, VkImageUsageFlags usage,
             VkImageCreateFlags flags = 0);
    bool build_images(std::string* why);
    bool build_descriptors(std::string* why);
    void dispatch_gamma(VkCommandBuffer cmd, int i, uint64_t n) const;
    void dispatch_delta(VkCommandBuffer cmd, int i, uint64_t n) const;

    const Pipelines* pipelines_ = nullptr;
    VkExtent2D extent_{}, flow_extent_{};
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    float flow_ = 0.5f;
    bool initialized_ = false;

    std::vector<Image*> all_;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize bytes_ = 0;
    VkBuffer ubo_ = VK_NULL_HANDLE;
    VkDeviceMemory ubo_memory_ = VK_NULL_HANDLE;
    VkDeviceSize ubo_stride_ = 0;
    VkSampler sampler_ = VK_NULL_HANDLE, border_sampler_ = VK_NULL_HANDLE, edge_sampler_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    uint32_t set_count_ = 0;

    Image frames_[2], mip_[kMipLevels], output_, dummy_;
    Alpha alpha_[kMipLevels];
    Beta beta_;
    Gamma gamma_[kMipLevels];
    Delta delta_[kDeltaInstances];
    VkDescriptorSet mip_sets_[2]{}, generate_sets_[2]{};
};

}  // namespace nvn::lsfg
