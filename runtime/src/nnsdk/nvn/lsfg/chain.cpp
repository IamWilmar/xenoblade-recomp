// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 lsfg-vk
// SPDX-License-Identifier: GPL-3.0-or-later

// Port de lsfg_common/mipmaps/alpha/beta/gamma/delta/generate/chain de Eden. Cambios respecto a Eden: Vulkan a
// pelo; una sola generación (2x) y un solo destino (nuestra imagen de salida); una imagen 1x1 a cero en vez de
// descriptores nulos; y una barrera de memoria global entre dispatches dependientes en vez de una por imagen
// (todas son de compute a compute: en el Adreno cuestan lo mismo).

#include "nnsdk/nvn/lsfg/chain.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "nnsdk/nvn/nvn.h"

namespace nvn::lsfg {

namespace {

constexpr VkFormat kColor = VK_FORMAT_R8G8B8A8_UNORM, kFlow = VK_FORMAT_R8_UNORM;
constexpr VkFormat kMotion = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkImageUsageFlags kChainUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

// Las constantes de cada pasada (el UBO de 48 bytes de Eden y lsfg-vk). SDR, ui_threshold 0.5 como Eden.
struct Constants {
    uint32_t input_offset[2];
    uint32_t first_iter, first_iter_s, advanced_color_kind, hdr_support;
    float resolution_inv_scale, timestamp, ui_threshold;
    uint32_t padding[3];
};
static_assert(sizeof(Constants) == 48);

// Las variantes que usa 2x: mipmaps (timestamp 0), el resto (0.5), gamma del primer nivel (first_iter) y delta
// del primero (first_iter_s).
enum Ubo { kUboMipmaps, kUboHalf, kUboFirstGamma, kUboFirstDelta, kUboCount };

// Lo que espera cada shader, en orden de binding: UBO, samplers, imágenes muestreadas, imágenes de storage.
struct Layout {
    uint32_t id;
    uint8_t ubo, samplers, sampled, storage;
};
constexpr Layout kLayouts[] = {
    {255, 1, 1, 1, 7}, {256, 1, 2, 5, 1},                                                          // mipmaps, generate
    {290, 0, 1, 1, 1}, {291, 0, 1, 1, 1}, {292, 0, 1, 1, 2}, {293, 0, 1, 2, 2},                    // alpha
    {298, 0, 1, 6, 2}, {299, 0, 1, 2, 2}, {300, 0, 1, 2, 2}, {301, 0, 1, 2, 2}, {302, 1, 1, 2, 6},  // beta
    {280, 1, 2, 5, 3}, {282, 0, 1, 3, 2}, {283, 0, 1, 2, 2}, {284, 0, 1, 2, 2}, {285, 1, 2, 4, 1},  // gamma
    {286, 0, 1, 3, 2}, {287, 0, 1, 2, 2}, {288, 0, 1, 2, 2}, {289, 1, 2, 4, 1}, {281, 1, 2, 6, 1},  // delta
    {294, 0, 1, 1, 1}, {295, 0, 1, 1, 1}, {296, 0, 1, 1, 1}, {297, 1, 2, 2, 1},
};

VkExtent2D halve(VkExtent2D e) { return {(e.width + 1) >> 1, (e.height + 1) >> 1}; }
VkExtent2D shifted(VkExtent2D e, int s) { return {e.width >> s, e.height >> s}; }
uint32_t groups(uint32_t size, uint32_t shift) { return (size + (1u << shift) - 1) >> shift; }

// sRGB de 8 bits: la cadena lee los valores codificados (como Lossless en Windows) por una vista UNORM.
VkFormat unorm_of(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        default: return f;
    }
}

uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags props) {
    const VkPhysicalDeviceMemoryProperties& m = vk()->memory;
    for (uint32_t k = 0; k < m.memoryTypeCount; k++)
        if ((bits & (1u << k)) && (m.memoryTypes[k].propertyFlags & props) == props) return k;
    return UINT32_MAX;
}

void barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
}

void bind_pipeline(VkCommandBuffer cmd, const Pass& p) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline); }
void bind_set(VkCommandBuffer cmd, const Pass& p, VkDescriptorSet set) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
}
void run(VkCommandBuffer cmd, const Pass& p, VkDescriptorSet set, VkExtent2D extent, uint32_t shift) {
    barrier(cmd);
    bind_pipeline(cmd, p);
    bind_set(cmd, p, set);
    vkCmdDispatch(cmd, groups(extent.width, shift), groups(extent.height, shift), 1);
}

// Escribe un descriptor set en el orden de los bindings de su pasada y comprueba los tipos contra el shader.
class Writer {
public:
    Writer(const Pass& pass, VkDescriptorSet set, VkImageView dummy) : pass_(pass), set_(set), dummy_(dummy) {}
    Writer& ubo(VkBuffer buffer, VkDeviceSize offset) {
        if (n_ >= kMax) return overflow();
        buffers_[n_] = {buffer, offset, sizeof(Constants)};
        writes_[n_] = write_for(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writes_[n_].pBufferInfo = &buffers_[n_];
        n_++;
        return *this;
    }
    Writer& sampler(VkSampler s) { return image(VK_DESCRIPTOR_TYPE_SAMPLER, s, VK_NULL_HANDLE); }
    // La entrada que falta en la primera iteración (nullptr) es la imagen vacía.
    Writer& optional(const Image* i) { return image(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_NULL_HANDLE, i ? i->view : dummy_); }
    Writer& sampled(const Image& i) { return image(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_NULL_HANDLE, i.view); }
    template <size_t N>
    Writer& sampled(const Image (&images)[N]) {
        for (const Image& i : images) sampled(i);
        return *this;
    }
    Writer& storage(const Image& i) { return image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, i.view); }
    template <size_t N>
    Writer& storage(const Image (&images)[N]) {
        for (const Image& i : images) storage(i);
        return *this;
    }
    bool write(std::string* why) {
        bool ok = !overflowed_ && n_ == pass_.types.size();
        for (uint32_t k = 0; ok && k < n_; k++) ok = writes_[k].descriptorType == pass_.types[k];
        if (!ok) {
            char buf[128];
            snprintf(buf, sizeof buf, "descriptores del shader %u mal puestos (%u para %zu bindings)", pass_.id, n_,
                     pass_.types.size());
            *why = buf;
            return false;
        }
        vkUpdateDescriptorSets(vk()->device, n_, writes_, 0, nullptr);
        return true;
    }

private:
    static constexpr uint32_t kMax = 16;
    Writer& overflow() {
        overflowed_ = true;
        return *this;
    }
    VkWriteDescriptorSet write_for(VkDescriptorType type) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set_;
        w.dstBinding = n_;
        w.descriptorCount = 1;
        w.descriptorType = type;
        return w;
    }
    Writer& image(VkDescriptorType type, VkSampler s, VkImageView view) {
        if (n_ >= kMax) return overflow();
        images_[n_] = {s, view, type == VK_DESCRIPTOR_TYPE_SAMPLER ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL};
        writes_[n_] = write_for(type);
        writes_[n_].pImageInfo = &images_[n_];
        n_++;
        return *this;
    }
    const Pass& pass_;
    VkDescriptorSet set_;
    VkImageView dummy_;
    uint32_t n_ = 0;
    bool overflowed_ = false;
    VkWriteDescriptorSet writes_[kMax];
    VkDescriptorImageInfo images_[kMax];
    VkDescriptorBufferInfo buffers_[kMax];
};

VkSampler make_sampler(VkSamplerAddressMode mode, VkCompareOp compare, bool white_border) {
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = mode;
    si.compareOp = compare;  // sin compareEnable: no se usa (como en Eden)
    si.maxLod = VK_LOD_CLAMP_NONE;
    si.borderColor = white_border ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VkSampler s = VK_NULL_HANDLE;
    vkCreateSampler(vk()->device, &si, nullptr, &s);
    return s;
}

}  // namespace

// --- Pipelines ---

bool Pipelines::create(const Modules& modules, bool float16, std::string* why) {
    destroy();
    VkDevice device = vk()->device;
    auto fail = [&](const char* format, auto... args) {
        char buf[192];
        snprintf(buf, sizeof buf, format, args...);
        *why = buf;
        destroy();
        return false;
    };
    for (const Layout& l : kLayouts) {
        auto it = modules.find(l.id);
        if (it == modules.end()) return fail("falta el shader %u", l.id);
        std::string entry, reason;
        std::vector<int> bindings;
        if (!audit_module(it->second, float16, &entry, &bindings, &reason))
            return fail("shader %u: %s", l.id, reason.c_str());
        Pass pass;
        pass.id = l.id;
        pass.types.insert(pass.types.end(), l.ubo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        pass.types.insert(pass.types.end(), l.samplers, VK_DESCRIPTOR_TYPE_SAMPLER);
        pass.types.insert(pass.types.end(), l.sampled, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        pass.types.insert(pass.types.end(), l.storage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        bool same = bindings.size() == pass.types.size();
        for (size_t k = 0; same && k < bindings.size(); k++) same = bindings[k] == pass.types[k];
        if (!same)
            return fail("shader %u: sus bindings no son los de la cadena (¿otra versión de Lossless?)", l.id);

        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = it->second.size() * 4;
        mi.pCode = it->second.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (VkResult r = vkCreateShaderModule(device, &mi, nullptr, &module); r != VK_SUCCESS)
            return fail("vkCreateShaderModule %u: %s", l.id, vk_result_name(r));

        std::vector<VkDescriptorSetLayoutBinding> b(pass.types.size());
        for (size_t k = 0; k < b.size(); k++)
            b[k] = {static_cast<uint32_t>(k), pass.types[k], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = static_cast<uint32_t>(b.size());
        li.pBindings = b.data();
        VkResult r = vkCreateDescriptorSetLayout(device, &li, nullptr, &pass.set_layout);
        if (r == VK_SUCCESS) {
            VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pi.setLayoutCount = 1;
            pi.pSetLayouts = &pass.set_layout;
            r = vkCreatePipelineLayout(device, &pi, nullptr, &pass.layout);
        }
        if (r == VK_SUCCESS) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = module;
            ci.stage.pName = entry.c_str();
            ci.layout = pass.layout;
            r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pass.pipeline);
        }
        vkDestroyShaderModule(device, module, nullptr);
        passes_[l.id] = pass;  // también si falló a medias: destroy() lo suelta
        if (r != VK_SUCCESS || !pass.pipeline)
            return fail("pipeline del shader %u: %s", l.id, vk_result_name(r));
    }
    return true;
}

void Pipelines::destroy() {
    VkDevice device = vk()->device;
    for (auto& [id, p] : passes_) {
        if (p.pipeline) vkDestroyPipeline(device, p.pipeline, nullptr);
        if (p.layout) vkDestroyPipelineLayout(device, p.layout, nullptr);
        if (p.set_layout) vkDestroyDescriptorSetLayout(device, p.set_layout, nullptr);
    }
    passes_.clear();
}

// --- Cadena ---

void Chain::add(Image& image, VkExtent2D extent, VkFormat format, VkFormat view_format, VkImageUsageFlags usage,
                VkImageCreateFlags flags) {
    image = Image{};
    image.extent = {std::max(1u, extent.width), std::max(1u, extent.height)};
    image.format = format;
    image.view_format = view_format;
    image.usage = usage;
    image.flags = flags;
    all_.push_back(&image);
}

bool Chain::create(const Pipelines& pipelines, VkExtent2D extent, VkFormat format, float flow, std::string* why) {
    destroy();
    pipelines_ = &pipelines;
    extent_ = extent;
    format_ = format;
    flow_ = flow;

    // Entradas: copia exacta de la textura de la ventana, en su formato (si es sRGB, leída por una vista UNORM).
    // Salida: si la ventana es sRGB, la imagen también (con vista UNORM para escribirla) para que el blit al
    // swapchain convierta el frame generado igual que convierte hoy el real.
    const VkFormat input_view = unorm_of(format);
    const bool srgb = input_view != format;
    for (Image& f : frames_)
        add(f, extent, format, input_view, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            srgb ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0);
    flow_extent_ = {std::max(1u, static_cast<uint32_t>(static_cast<float>(extent.width) * flow)),
                    std::max(1u, static_cast<uint32_t>(static_cast<float>(extent.height) * flow))};
    for (int i = 0; i < kMipLevels; i++) add(mip_[i], shifted(flow_extent_, i), kFlow, kFlow, kChainUsage);
    for (int l = 0; l < kMipLevels; l++) {
        Alpha& a = alpha_[l];
        const VkExtent2D half = halve(mip_[l].extent), quarter = halve(halve(mip_[l].extent));
        add(a.temp1, half, kColor, kColor, kChainUsage);
        add(a.temp2, half, kColor, kColor, kChainUsage);
        for (int i = 0; i < 2; i++) {
            add(a.temp3[i], quarter, kColor, kColor, kChainUsage);
            for (int h = 0; h < kHistory; h++) add(a.out[h][i], quarter, kColor, kColor, kChainUsage);
        }
    }
    {
        const VkExtent2D e = alpha_[0].out[0][0].extent;
        for (int i = 0; i < 2; i++) {
            add(beta_.temp1[i], e, kColor, kColor, kChainUsage);
            add(beta_.temp2[i], e, kColor, kColor, kChainUsage);
        }
        for (int i = 0; i < kBetaOutputs; i++) add(beta_.out[i], shifted(e, i), kFlow, kFlow, kChainUsage);
    }
    for (int i = 0; i < kMipLevels; i++) {
        Gamma& g = gamma_[i];
        const VkExtent2D e = alpha_[kMipLevels - 1 - i].out[0][0].extent;
        for (Image& t : g.temp1) add(t, e, kColor, kColor, kChainUsage);
        for (Image& t : g.temp2) add(t, e, kColor, kColor, kChainUsage);
        add(g.out, e, kMotion, kMotion, kChainUsage);
    }
    for (int j = 0; j < kDeltaInstances; j++) {
        Delta& d = delta_[j];
        const VkExtent2D e = alpha_[kMipLevels - 1 - (kFirstDeltaLevel + j)].out[0][0].extent;
        for (Image& t : d.temp1) add(t, e, kColor, kColor, kChainUsage);
        for (Image& t : d.temp2) add(t, e, kColor, kColor, kChainUsage);
        add(d.out1, e, kMotion, kMotion, kChainUsage);
        add(d.out2, e, kMotion, kMotion, kChainUsage);
    }
    if (srgb)
        add(output_, extent, VK_FORMAT_R8G8B8A8_SRGB, kColor, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
    else
        add(output_, extent, kColor, kColor, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);  // muestreada: FSR al presentar
    add(dummy_, {1, 1}, kMotion, kMotion, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    if (!build_images(why) || !build_descriptors(why)) {
        destroy();
        return false;
    }
    return true;
}

bool Chain::build_images(std::string* why) {
    VkContext* c = vk();
    auto fail = [&](std::string reason) {
        *why = std::move(reason);
        return false;
    };
    char buf[160];
    auto has = [&](VkFormat f, VkFormatFeatureFlags need) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(c->gpu, f, &fp);
        return (fp.optimalTilingFeatures & need) == need;
    };
    for (Image* i : all_) {
        VkFormatFeatureFlags view_need = 0, image_need = 0;
        if (i->usage & VK_IMAGE_USAGE_SAMPLED_BIT)
            view_need |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        if (i->usage & VK_IMAGE_USAGE_STORAGE_BIT) view_need |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        if (i == &output_) image_need |= VK_FORMAT_FEATURE_BLIT_SRC_BIT;
        if (i->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) image_need |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if (!has(i->view_format, view_need) || !has(i->format, image_need)) {
            snprintf(buf, sizeof buf, "el formato %d (vista %d) no sirve para la cadena", i->format, i->view_format);
            *why = buf;
            return false;
        }
    }
    uint32_t bits = ~0u;
    bytes_ = 0;
    for (Image* i : all_) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.flags = i->flags;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = i->format;
        ii.extent = {i->extent.width, i->extent.height, 1};
        ii.mipLevels = ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = i->usage;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (VkResult r = vkCreateImage(c->device, &ii, nullptr, &i->image); r != VK_SUCCESS) {
            snprintf(buf, sizeof buf, "vkCreateImage %ux%u formato %d: %s", i->extent.width, i->extent.height, i->format,
                     vk_result_name(r));
            *why = buf;
            return false;
        }
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(c->device, i->image, &req);
        bits &= req.memoryTypeBits;
        i->offset = (bytes_ + req.alignment - 1) / req.alignment * req.alignment;
        bytes_ = i->offset + req.size;
    }
    // Todas en una reserva: el driver del Adreno lleva mal muchas reservas pequeñas (gpu_memory.h).
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = bytes_;
    ai.memoryTypeIndex = find_memory(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX) return fail("no hay un tipo de memoria para todas las imágenes");
    if (VkResult r = vkAllocateMemory(c->device, &ai, nullptr, &memory_); r != VK_SUCCESS) {
        snprintf(buf, sizeof buf, "vkAllocateMemory de %llu KB: %s", (unsigned long long)(bytes_ >> 10), vk_result_name(r));
        *why = buf;
        return false;
    }
    for (Image* i : all_) {
        if (VkResult r = vkBindImageMemory(c->device, i->image, memory_, i->offset); r != VK_SUCCESS)
            return fail(std::string("vkBindImageMemory: ") + vk_result_name(r));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        usage.usage = i->usage & (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (i->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) vi.pNext = &usage;  // la vista UNORM solo es de storage
        vi.image = i->image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = i->view_format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (VkResult r = vkCreateImageView(c->device, &vi, nullptr, &i->view); r != VK_SUCCESS)
            return fail(std::string("vkCreateImageView: ") + vk_result_name(r));
    }

    // Constantes: una variante por bloque alineado, escritas una vez.
    const VkDeviceSize align = std::max<VkDeviceSize>(c->props.limits.minUniformBufferOffsetAlignment, 16);
    ubo_stride_ = (sizeof(Constants) + align - 1) / align * align;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = ubo_stride_ * kUboCount;
    bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (VkResult r = vkCreateBuffer(c->device, &bi, nullptr, &ubo_); r != VK_SUCCESS)
        return fail(std::string("vkCreateBuffer: ") + vk_result_name(r));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, ubo_, &req);
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped = nullptr;
    if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(c->device, &ai, nullptr, &ubo_memory_) != VK_SUCCESS ||
        vkBindBufferMemory(c->device, ubo_, ubo_memory_, 0) != VK_SUCCESS ||
        vkMapMemory(c->device, ubo_memory_, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
        return fail("sin memoria para las constantes");
    for (int v = 0; v < kUboCount; v++) {
        Constants k{};
        k.first_iter = v == kUboFirstGamma;
        k.first_iter_s = v == kUboFirstDelta;
        k.resolution_inv_scale = 1.0f / flow_;
        k.timestamp = v == kUboMipmaps ? 0.0f : 0.5f;
        k.ui_threshold = 0.5f;
        memcpy(static_cast<uint8_t*>(mapped) + v * ubo_stride_, &k, sizeof k);
    }
    vkUnmapMemory(c->device, ubo_memory_);

    sampler_ = make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_COMPARE_OP_NEVER, false);
    border_sampler_ = make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_COMPARE_OP_NEVER, true);
    edge_sampler_ = make_sampler(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_COMPARE_OP_ALWAYS, false);
    if (!sampler_ || !border_sampler_ || !edge_sampler_) return fail("vkCreateSampler");
    return true;
}

bool Chain::build_descriptors(std::string* why) {
    VkContext* c = vk();
    auto fail = [&](std::string reason) {
        *why = std::move(reason);
        return false;
    };
    const Pipelines& p = *pipelines_;
    using namespace shader;

    // Todos los sets de una vez: primero qué set es de qué pasada, luego el pool justo y la reserva.
    struct Request {
        VkDescriptorSet* set;
        const Pass* pass;
    };
    std::vector<Request> requests;
    for (VkDescriptorSet& s : mip_sets_) requests.push_back({&s, &p.get(kMipmaps)});
    for (Alpha& a : alpha_) {
        for (int s = 0; s < kAlphaStages - 1; s++) requests.push_back({&a.sets[s], &p.get(kAlpha[s])});
        for (VkDescriptorSet& s : a.last) requests.push_back({&s, &p.get(kAlpha[3])});
    }
    for (VkDescriptorSet& s : beta_.first) requests.push_back({&s, &p.get(kBeta[0])});
    for (int s = 0; s < kBetaStages - 1; s++) requests.push_back({&beta_.sets[s], &p.get(kBeta[s + 1])});
    for (Gamma& g : gamma_) {
        for (VkDescriptorSet& s : g.first) requests.push_back({&s, &p.get(kGamma[0])});
        for (int s = 0; s < kGammaStages - 1; s++) requests.push_back({&g.sets[s], &p.get(kGamma[s + 1])});
    }
    for (Delta& d : delta_) {
        for (VkDescriptorSet& s : d.first) requests.push_back({&s, &p.get(kDelta[0])});
        for (int s = 0; s < 4; s++) requests.push_back({&d.sets[s], &p.get(kDelta[s + 1])});
        for (VkDescriptorSet& s : d.sixth) requests.push_back({&s, &p.get(kDelta[5])});
        for (int s = 4; s < kDeltaStages - 2; s++) requests.push_back({&d.sets[s], &p.get(kDelta[s + 2])});
    }
    for (VkDescriptorSet& s : generate_sets_) requests.push_back({&s, &p.get(kGenerate)});

    const VkDescriptorType kinds[] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkDescriptorPoolSize sizes[4];
    std::vector<VkDescriptorSetLayout> layouts;
    for (int k = 0; k < 4; k++) sizes[k] = {kinds[k], 0};
    for (const Request& r : requests) {
        layouts.push_back(r.pass->set_layout);
        for (VkDescriptorType t : r.pass->types)
            for (VkDescriptorPoolSize& s : sizes)
                if (s.type == t) s.descriptorCount++;
    }
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = static_cast<uint32_t>(requests.size());
    pi.poolSizeCount = 4;
    pi.pPoolSizes = sizes;
    if (VkResult r = vkCreateDescriptorPool(c->device, &pi, nullptr, &pool_); r != VK_SUCCESS)
        return fail(std::string("vkCreateDescriptorPool: ") + vk_result_name(r));
    std::vector<VkDescriptorSet> sets(requests.size());
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool_;
    dai.descriptorSetCount = static_cast<uint32_t>(sets.size());
    dai.pSetLayouts = layouts.data();
    if (VkResult r = vkAllocateDescriptorSets(c->device, &dai, sets.data()); r != VK_SUCCESS)
        return fail(std::string("vkAllocateDescriptorSets: ") + vk_result_name(r));
    for (size_t k = 0; k < requests.size(); k++) *requests[k].set = sets[k];
    set_count_ = static_cast<uint32_t>(sets.size());

    const VkDeviceSize mip_ubo = kUboMipmaps * ubo_stride_, half_ubo = kUboHalf * ubo_stride_;
    const VkImageView dummy = dummy_.view;
    auto w = [&](uint32_t id, VkDescriptorSet set) { return Writer(p.get(id), set, dummy); };
    bool ok = true;

    for (int i = 0; i < 2 && ok; i++)
        ok = w(kMipmaps, mip_sets_[i]).ubo(ubo_, mip_ubo).sampler(sampler_).sampled(frames_[i]).storage(mip_).write(why);

    for (int l = 0; l < kMipLevels && ok; l++) {
        Alpha& a = alpha_[l];
        ok = w(kAlpha[0], a.sets[0]).sampler(sampler_).sampled(mip_[l]).storage(a.temp1).write(why) &&
             w(kAlpha[1], a.sets[1]).sampler(sampler_).sampled(a.temp1).storage(a.temp2).write(why) &&
             w(kAlpha[2], a.sets[2]).sampler(sampler_).sampled(a.temp2).storage(a.temp3).write(why);
        for (int h = 0; h < kHistory && ok; h++)
            ok = w(kAlpha[3], a.last[h]).sampler(sampler_).sampled(a.temp3).storage(a.out[h]).write(why);
    }

    {  // beta lee la historia de alpha del nivel 0: frames n-2, n-1 y n
        const auto& in = alpha_[0].out;
        for (int h = 0; h < kHistory && ok; h++)
            ok = w(kBeta[0], beta_.first[h])
                     .sampler(border_sampler_)
                     .sampled(in[(h + 1) % kHistory])
                     .sampled(in[(h + 2) % kHistory])
                     .sampled(in[h])
                     .storage(beta_.temp1)
                     .write(why);
        ok = ok && w(kBeta[1], beta_.sets[0]).sampler(sampler_).sampled(beta_.temp1).storage(beta_.temp2).write(why) &&
             w(kBeta[2], beta_.sets[1]).sampler(sampler_).sampled(beta_.temp2).storage(beta_.temp1).write(why) &&
             w(kBeta[3], beta_.sets[2]).sampler(sampler_).sampled(beta_.temp1).storage(beta_.temp2).write(why) &&
             w(kBeta[4], beta_.sets[3]).ubo(ubo_, half_ubo).sampler(sampler_).sampled(beta_.temp2).storage(beta_.out).write(why);
    }

    for (int i = 0; i < kMipLevels && ok; i++) {  // gamma del nivel más pequeño al más grande
        Gamma& g = gamma_[i];
        const int level = kMipLevels - 1 - i;
        const auto& in = alpha_[level].out;
        const Image* previous = i ? &gamma_[i - 1].out : nullptr;
        const Image& flow = beta_.out[std::min(level, kBetaOutputs - 1)];
        const VkDeviceSize ubo = (i ? kUboHalf : kUboFirstGamma) * ubo_stride_;
        for (int h = 0; h < kHistory && ok; h++)
            ok = w(kGamma[0], g.first[h])
                     .ubo(ubo_, ubo)
                     .sampler(border_sampler_)
                     .sampler(edge_sampler_)
                     .sampled(in[(h + 2) % kHistory])
                     .sampled(in[h])
                     .optional(previous)
                     .storage(g.temp1)
                     .write(why);
        ok = ok && w(kGamma[1], g.sets[0]).sampler(sampler_).sampled(g.temp1).storage(g.temp2).write(why) &&
             w(kGamma[2], g.sets[1]).sampler(sampler_).sampled(g.temp2).storage(g.temp1[0]).storage(g.temp1[1]).write(why) &&
             w(kGamma[3], g.sets[2]).sampler(sampler_).sampled(g.temp1[0]).sampled(g.temp1[1]).storage(g.temp2).write(why) &&
             w(kGamma[4], g.sets[3])
                 .ubo(ubo_, ubo)
                 .sampler(sampler_)
                 .sampler(edge_sampler_)
                 .sampled(g.temp2)
                 .optional(previous)
                 .sampled(flow)
                 .storage(g.out)
                 .write(why);
    }

    for (int j = 0; j < kDeltaInstances && ok; j++) {  // delta en los tres niveles más grandes
        Delta& d = delta_[j];
        const int i = kFirstDeltaLevel + j, level = kMipLevels - 1 - i;
        const auto& in = alpha_[level].out;
        const Image* previous_gamma = j ? &gamma_[i - 1].out : nullptr;
        const Image* previous1 = j ? &delta_[j - 1].out1 : nullptr;
        const Image* previous2 = j ? &delta_[j - 1].out2 : nullptr;
        const Image& flow = beta_.out[level];
        const VkDeviceSize ubo = (j ? kUboHalf : kUboFirstDelta) * ubo_stride_;
        for (int h = 0; h < kHistory && ok; h++)
            ok = w(kDelta[0], d.first[h])
                     .ubo(ubo_, ubo)
                     .sampler(border_sampler_)
                     .sampler(edge_sampler_)
                     .sampled(in[(h + 2) % kHistory])
                     .sampled(in[h])
                     .optional(previous_gamma)
                     .storage(d.temp1)
                     .write(why) &&
                 w(kDelta[5], d.sixth[h])
                     .ubo(ubo_, ubo)
                     .sampler(border_sampler_)
                     .sampler(edge_sampler_)
                     .sampled(in[(h + 2) % kHistory])
                     .sampled(in[h])
                     .optional(previous_gamma)
                     .optional(previous1)
                     .storage(d.temp2[0])
                     .write(why);
        ok = ok && w(kDelta[1], d.sets[0]).sampler(sampler_).sampled(d.temp1).storage(d.temp2).write(why) &&
             w(kDelta[2], d.sets[1]).sampler(sampler_).sampled(d.temp2).storage(d.temp1[0]).storage(d.temp1[1]).write(why) &&
             w(kDelta[3], d.sets[2]).sampler(sampler_).sampled(d.temp1[0]).sampled(d.temp1[1]).storage(d.temp2).write(why) &&
             w(kDelta[4], d.sets[3])
                 .ubo(ubo_, ubo)
                 .sampler(sampler_)
                 .sampler(edge_sampler_)
                 .sampled(d.temp2)
                 .optional(previous_gamma)
                 .sampled(flow)
                 .storage(d.out1)
                 .write(why) &&
             w(kDelta[6], d.sets[4]).sampler(sampler_).sampled(d.temp2[0]).storage(d.temp1[0]).write(why) &&
             w(kDelta[7], d.sets[5]).sampler(sampler_).sampled(d.temp1[0]).storage(d.temp2[0]).write(why) &&
             w(kDelta[8], d.sets[6]).sampler(sampler_).sampled(d.temp2[0]).storage(d.temp1[0]).write(why) &&
             w(kDelta[9], d.sets[7])
                 .ubo(ubo_, ubo)
                 .sampler(sampler_)
                 .sampler(edge_sampler_)
                 .sampled(d.temp1[0])
                 .optional(previous2)
                 .storage(d.out2)
                 .write(why);
    }

    // generate: el set i (frame n con n % 2 == i) lee el anterior (frames_[1 - i]) y el actual (frames_[i]).
    for (int i = 0; i < 2 && ok; i++)
        ok = w(kGenerate, generate_sets_[i])
                 .ubo(ubo_, half_ubo)
                 .sampler(sampler_)
                 .sampler(edge_sampler_)
                 .sampled(frames_[1 - i])
                 .sampled(frames_[i])
                 .sampled(gamma_[kMipLevels - 1].out)
                 .sampled(delta_[kDeltaInstances - 1].out1)
                 .sampled(delta_[kDeltaInstances - 1].out2)
                 .storage(output_)
                 .write(why);
    return ok;
}

void Chain::destroy() {
    VkContext* c = vk();
    if (!c) return;
    for (Image* i : all_) {
        if (i->view) vkDestroyImageView(c->device, i->view, nullptr);
        if (i->image) vkDestroyImage(c->device, i->image, nullptr);
        *i = Image{};
    }
    all_.clear();
    if (memory_) vkFreeMemory(c->device, memory_, nullptr);
    if (ubo_) vkDestroyBuffer(c->device, ubo_, nullptr);
    if (ubo_memory_) vkFreeMemory(c->device, ubo_memory_, nullptr);
    for (VkSampler s : {sampler_, border_sampler_, edge_sampler_})
        if (s) vkDestroySampler(c->device, s, nullptr);
    if (pool_) vkDestroyDescriptorPool(c->device, pool_, nullptr);  // con él, sus sets
    memory_ = ubo_memory_ = VK_NULL_HANDLE;
    ubo_ = VK_NULL_HANDLE;
    sampler_ = border_sampler_ = edge_sampler_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    bytes_ = 0;
    set_count_ = 0;
    initialized_ = false;
}

void Chain::record_init(VkCommandBuffer cmd) {
    std::vector<VkImageMemoryBarrier> b(all_.size());  // una vez por cadena
    for (size_t k = 0; k < all_.size(); k++) {
        b[k] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_TRANSFER_WRITE_BIT;
        b[k].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b[k].srcQueueFamilyIndex = b[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[k].image = all_[k]->image;
        b[k].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                         static_cast<uint32_t>(b.size()), b.data());
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, dummy_.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    initialized_ = true;
}

void Chain::record_shared(VkCommandBuffer cmd, uint64_t n) const {
    const Pipelines& p = *pipelines_;
    using namespace shader;
    run(cmd, p.get(kMipmaps), mip_sets_[n % 2], flow_extent_, 6);
    for (int stage = 0; stage < kAlphaStages; stage++) {  // cada etapa en los siete niveles, con una barrera
        const Pass& pass = p.get(kAlpha[stage]);
        barrier(cmd);
        bind_pipeline(cmd, pass);
        for (const Alpha& a : alpha_) {
            const VkExtent2D e = stage < 2 ? a.temp1.extent : a.temp3[0].extent;
            bind_set(cmd, pass, stage < kAlphaStages - 1 ? a.sets[stage] : a.last[n % kHistory]);
            vkCmdDispatch(cmd, groups(e.width, 3), groups(e.height, 3), 1);
        }
    }
    const VkExtent2D e = beta_.temp1[0].extent;
    run(cmd, p.get(kBeta[0]), beta_.first[n % kHistory], e, 3);
    for (int s = 1; s < kBetaStages; s++) run(cmd, p.get(kBeta[s]), beta_.sets[s - 1], e, s == kBetaStages - 1 ? 5 : 3);
}

void Chain::dispatch_gamma(VkCommandBuffer cmd, int i, uint64_t n) const {
    const Pipelines& p = *pipelines_;
    const Gamma& g = gamma_[i];
    const VkExtent2D e = g.temp1[0].extent;
    run(cmd, p.get(shader::kGamma[0]), g.first[n % kHistory], e, 3);
    for (int s = 1; s < kGammaStages; s++) run(cmd, p.get(shader::kGamma[s]), g.sets[s - 1], e, 3);
}

void Chain::dispatch_delta(VkCommandBuffer cmd, int j, uint64_t n) const {
    const Pipelines& p = *pipelines_;
    const Delta& d = delta_[j];
    const VkExtent2D e = d.temp1[0].extent;
    run(cmd, p.get(shader::kDelta[0]), d.first[n % kHistory], e, 3);
    for (int s = 1; s <= 4; s++) run(cmd, p.get(shader::kDelta[s]), d.sets[s - 1], e, 3);
    run(cmd, p.get(shader::kDelta[5]), d.sixth[n % kHistory], e, 3);
    for (int s = 6; s < kDeltaStages; s++) run(cmd, p.get(shader::kDelta[s]), d.sets[s - 2], e, 3);
}

void Chain::record_generate(VkCommandBuffer cmd, uint64_t n) const {
    for (int i = 0; i < kMipLevels; i++) {
        dispatch_gamma(cmd, i, n);
        if (i >= kFirstDeltaLevel) dispatch_delta(cmd, i - kFirstDeltaLevel, n);
    }
    run(cmd, pipelines_->get(shader::kGenerate), generate_sets_[n % 2], output_.extent, 4);
}

}  // namespace nvn::lsfg
