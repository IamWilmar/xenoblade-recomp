// FSR 1 al presentar (ver fsr.h). Todo en el hilo que presenta, con executor_mutex() tomado por quien llama.

#include "nnsdk/nvn/fsr.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/shaders/spirv.h"

namespace nvn::fsr {
namespace {

uint32_t as_uint(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags props) {
    const VkPhysicalDeviceMemoryProperties& m = vk()->memory;
    for (uint32_t k = 0; k < m.memoryTypeCount; k++)
        if ((bits & (1u << k)) && (m.memoryTypes[k].propertyFlags & props) == props) return k;
    return UINT32_MAX;
}

class Fsr {
public:
    explicit Fsr(const char* who) : who_(who) {}

    bool record(VkCommandBuffer cmd, VkImageView src, VkExtent2D in, VkExtent2D viewport, VkImage dst, VkFormat format,
                VkExtent2D out, bool easu) {
        if (failed_ || !src || format != VK_FORMAT_R8G8B8A8_UNORM || NVN_FLAG("sin_fsr")) return false;
        if (!setup() || !images(out)) return false;
        if (!easu && !rcas_up_.pipeline) return false;  // sin la pasada única: quien llama hace el blit

        // Lo de antes en la cola (el frame del juego, el FSR del frame anterior que leía estas imágenes) antes que esto.
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);

        // EASU: de src (in) a mid_ (out).
        const float iw = static_cast<float>(in.width), ih = static_cast<float>(in.height);
        const float vw = static_cast<float>(viewport.width), vh = static_cast<float>(viewport.height);
        const float ow = static_cast<float>(out.width), oh = static_cast<float>(out.height);
        uint32_t easu_con[16];
        easu_constants(in, viewport, out, easu_con);
        VkMemoryBarrier rw{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        rw.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        // La nitidez de RCAS en "stops" (0 = la máxima; 0,25 un poco menos, sin halos).
        if (easu) {
            dispatch(cmd, easu_, src, mid_.view, easu_con, sizeof easu_con, out);
            rw.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &rw, 0,
                                 nullptr, 0, nullptr);
            // RCAS: de mid_ a out_.
            const uint32_t rcas[4] = {as_uint(std::exp2(-0.25f)), 0, 0, 0};
            dispatch(cmd, rcas_, mid_.view, out_.view, rcas, sizeof rcas, out);
        } else {
            // Sin EASU: RCAS de src a out_, leyendo src con el filtro lineal en el centro de cada píxel de salida (con1: el uv
            // de src por píxel de salida y el máximo, medio texel dentro de la zona que se escala).
            const uint32_t rcas[8] = {as_uint(std::exp2(-0.25f)), 0, 0, 0,
                                      as_uint(vw / ow / iw), as_uint(vh / oh / ih), as_uint((vw - 0.5f) / iw), as_uint((vh - 0.5f) / ih)};
            dispatch(cmd, rcas_up_, src, out_.view, rcas, sizeof rcas, out);
        }
        rw.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &rw, 0, nullptr,
                             0, nullptr);

        VkImageCopy region{};
        region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {out.width, out.height, 1};
        vkCmdCopyImage(cmd, out_.image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        return true;
    }

    // Ver fsr.h (record_scene). Solo mid_ (si rcas), del tamaño de dst; dst es del ejecutor (una Texture suya).
    bool scene(VkCommandBuffer cmd, VkImageView src, VkExtent2D in, VkImageView dst, VkExtent2D out, bool rcas) {
        if (failed_ || !src || !dst || !in.width || !in.height || !out.width || !out.height) return false;
        if (!setup()) return false;
        if (rcas && !mid_image(out)) rcas = false;  // sin su intermedia: solo EASU
        VkMemoryBarrier mb{};  // lo que pintó src y lo que leía dst (el frame anterior)
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        uint32_t easu_con[16];
        easu_constants(in, in, out, easu_con);  // la textura entera: el juego la estira con uv de 0 a 1
        dispatch(cmd, easu_, src, rcas ? mid_.view : dst, easu_con, sizeof easu_con, out);
        VkMemoryBarrier rw{};
        rw.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        rw.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        if (rcas) {
            rw.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &rw, 0,
                                 nullptr, 0, nullptr);
            const uint32_t con[4] = {as_uint(std::exp2(-0.25f)), 0, 0, 0};
            dispatch(cmd, rcas_, mid_.view, dst, con, sizeof con, out);
        }
        // Lo escrito, antes del draw que lo muestrea (y de cualquier otra cosa que venga detrás).
        rw.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &rw, 0, nullptr,
                             0, nullptr);
        return true;
    }

    // Ver fsr.h. raw_: la copia UNORM de la textura de la ventana, de su tamaño (se crea una vez, la primera vez que hace falta).
    bool blit_encoded(VkCommandBuffer cmd, VkImage src, VkFormat raw_format, VkExtent2D in, VkRect2D rect, VkImage dst,
                      VkExtent2D out) {
        if (raw_failed_ || !src || !rect.extent.width || !rect.extent.height || rect.offset.x < 0 || rect.offset.y < 0 ||
            rect.offset.x + rect.extent.width > in.width || rect.offset.y + rect.extent.height > in.height)
            return false;
        if (!raw_.image) {
            if (!image(raw_, in, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       raw_format)) {
                NVN_LOG("presentación: no se creó la copia UNORM de la ventana sRGB; blit directo (decodifica el sRGB)\n");
                raw_failed_ = true;
                return false;
            }
            raw_extent_ = in;
            raw_format_ = raw_format;
            raw_new_ = true;
            NVN_LOG("presentación: ventana sRGB sin FSR: blit desde una copia UNORM %ux%u (formato %d)\n", in.width, in.height,
                    static_cast<int>(raw_format));
        } else if (raw_extent_.width != in.width || raw_extent_.height != in.height || raw_format_ != raw_format) {
            NVN_LOG("presentación: la textura de la ventana cambió de tamaño o formato; blit directo (decodifica el sRGB)\n");
            raw_failed_ = true;
            return false;
        }
        // Lo de antes en la cola (el blit del frame anterior, que leía raw_) antes que esta copia; la primera vez, su layout.
        VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ib.srcAccessMask = raw_new_ ? 0 : VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ib.oldLayout = raw_new_ ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = raw_.image;
        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &ib);
        raw_new_ = false;
        // Un texel más a la derecha y abajo si lo hay: el filtro lineal del blit puede leerlo en el borde de la zona, como lo
        // leía de la textura.
        VkImageCopy copy{};
        copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.srcOffset = copy.dstOffset = {rect.offset.x, rect.offset.y, 0};
        copy.extent = {std::min(rect.extent.width + 1, in.width - static_cast<uint32_t>(rect.offset.x)),
                       std::min(rect.extent.height + 1, in.height - static_cast<uint32_t>(rect.offset.y)), 1};
        vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, raw_.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
        VkImageBlit b{};
        b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[0] = {rect.offset.x, rect.offset.y, 0};
        b.srcOffsets[1] = {rect.offset.x + static_cast<int32_t>(rect.extent.width),
                           rect.offset.y + static_cast<int32_t>(rect.extent.height), 1};
        b.dstOffsets[1] = {static_cast<int32_t>(out.width), static_cast<int32_t>(out.height), 1};
        vkCmdBlitImage(cmd, raw_.image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                       VK_FILTER_LINEAR);
        return true;
    }

private:
    // Las constantes de EASU, como FsrEasuCon de ffx_fsr1.h: con0 va con la zona que se escala (viewport) y el resto con
    // el tamaño de la textura (in).
    static void easu_constants(VkExtent2D in, VkExtent2D viewport, VkExtent2D out, uint32_t con[16]) {
        const float iw = static_cast<float>(in.width), ih = static_cast<float>(in.height);
        const float vw = static_cast<float>(viewport.width), vh = static_cast<float>(viewport.height);
        const float ow = static_cast<float>(out.width), oh = static_cast<float>(out.height);
        const uint32_t c[16] = {as_uint(vw / ow), as_uint(vh / oh), as_uint(0.5f * vw / ow - 0.5f), as_uint(0.5f * vh / oh - 0.5f),
                                as_uint(1.0f / iw), as_uint(1.0f / ih), as_uint(1.0f / iw), as_uint(-1.0f / ih),
                                as_uint(-1.0f / iw), as_uint(2.0f / ih), as_uint(1.0f / iw), as_uint(2.0f / ih),
                                as_uint(0.0f), as_uint(4.0f / ih), 0, 0};
        memcpy(con, c, sizeof c);
    }

    bool is_scene() const { return strcmp(who_, "fsr") != 0; }

    // FSR de la escena: la intermedia de EASU a RCAS, del tamaño de la salida. Si la salida cambia de tamaño no se
    // rehace (la de antes puede estar aún en la GPU): ese draw va sin RCAS.
    bool mid_image(VkExtent2D out) {
        if (mid_.image) return extent_.width == out.width && extent_.height == out.height;
        if (!image(mid_, out, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
            NVN_LOG("%s: no se creó su imagen intermedia %ux%u; sin RCAS\n", who_, out.width, out.height);
            mid_ = Image{};
            return false;
        }
        extent_ = out;
        pending_layout_ = true;
        return true;
    }

    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    struct Pass {
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    bool fail(const char* why) {
        NVN_LOG("%s: %s; %s\n", who_, why, is_scene() ? "el juego estira la escena como siempre" : "se escala con el blit de siempre");
        failed_ = true;
        return false;
    }

    bool setup() {
        if (layout_) return true;
        VkContext* c = vk();
        if (!c->push_descriptor) return fail("sin VK_KHR_push_descriptor");
        push_ = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR"));
        if (!push_) return fail("sin vkCmdPushDescriptorSetKHR");
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(c->device, &si, nullptr, &sampler_) != VK_SUCCESS) return fail("vkCreateSampler");
        const VkDescriptorSetLayoutBinding b[2] = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dl.bindingCount = 2;
        dl.pBindings = b;
        if (vkCreateDescriptorSetLayout(c->device, &dl, nullptr, &set_layout_) != VK_SUCCESS)
            return fail("vkCreateDescriptorSetLayout");
        const VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &set_layout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pc;
        if (vkCreatePipelineLayout(c->device, &pl, nullptr, &layout_) != VK_SUCCESS) return fail("vkCreatePipelineLayout");
        // En FP16 si el dispositivo lo tiene (shaderFloat16 y shaderInt16, que vk.cpp enciende si están): el Adreno 650 hace
        // el doble de FP16 por ciclo y el EASU es casi todo aritmética. <datos>/fsr_fp32 (al arrancar): los de FP32.
        bool half = vk()->shader_float16 && !NVN_FLAG("fsr_fp32");
        if (half && !(pipeline(easu_, k_fsr_easu_h_spv, sizeof k_fsr_easu_h_spv) &&
                      pipeline(rcas_, k_fsr_rcas_h_spv, sizeof k_fsr_rcas_h_spv)))
            half = false;
        if (!half &&
            (!pipeline(easu_, k_fsr_easu_spv, sizeof k_fsr_easu_spv) || !pipeline(rcas_, k_fsr_rcas_spv, sizeof k_fsr_rcas_spv)))
            return fail("no se crearon sus pipelines");
        // La pasada única sin EASU (opcional: sin ella, ese nivel es el blit).
        if (!pipeline(rcas_up_, half ? k_fsr_rcas_up_h_spv : k_fsr_rcas_up_spv,
                      half ? sizeof k_fsr_rcas_up_h_spv : sizeof k_fsr_rcas_up_spv))
            rcas_up_.pipeline = VK_NULL_HANDLE;
        NVN_LOG("%s: listo (EASU + RCAS, %s%s)\n", who_, half ? "FP16" : "FP32", rcas_up_.pipeline ? "; con la pasada única de RCAS" : "");
        return true;
    }

    bool pipeline(Pass& pass, const uint32_t* code, size_t size) {
        VkContext* c = vk();
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = size;
        mi.pCode = code;
        VkShaderModule module;
        if (vkCreateShaderModule(c->device, &mi, nullptr, &module) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "main";
        ci.layout = layout_;
        const VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, nullptr, &pass.pipeline);
        vkDestroyShaderModule(c->device, module, nullptr);
        return r == VK_SUCCESS;
    }

    bool image(Image& img, VkExtent2D extent, VkImageUsageFlags usage, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM) {
        VkContext* c = vk();
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = format;
        ii.extent = {extent.width, extent.height, 1};
        ii.mipLevels = ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = usage;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(c->device, &ii, nullptr, &img.image) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(c->device, img.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(c->device, &ai, nullptr, &img.memory) != VK_SUCCESS ||
            vkBindImageMemory(c->device, img.image, img.memory, 0) != VK_SUCCESS)
            return false;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return vkCreateImageView(c->device, &vi, nullptr, &img.view) == VK_SUCCESS;
    }

    // Las dos imágenes intermedias, del tamaño de la pantalla (se crean una vez: el swapchain no cambia de tamaño).
    bool images(VkExtent2D out) {
        if (mid_.image && extent_.width == out.width && extent_.height == out.height) return true;
        if (mid_.image) return fail("el swapchain cambió de tamaño");
        if (!image(mid_, out, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) ||
            !image(out_, out, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
            return fail("no se crearon sus imágenes");
        extent_ = out;
        pending_layout_ = true;
        return true;
    }

    void dispatch(VkCommandBuffer cmd, const Pass& pass, VkImageView src, VkImageView dst, const void* constants,
                  uint32_t size, VkExtent2D out) {
        if (pending_layout_) {  // la primera vez: de UNDEFINED a GENERAL
            pending_layout_ = false;
            // (las que haya: la del FSR de la escena solo tiene mid_)
            VkImageMemoryBarrier b[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
            uint32_t n = 0;
            for (VkImage img : {mid_.image, out_.image}) {
                if (!img) continue;
                b[n].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                b[n].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b[n].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b[n].srcQueueFamilyIndex = b[n].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b[n].image = img;
                b[n].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                n++;
            }
            if (n)
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                     0, nullptr, n, b);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
        VkDescriptorImageInfo in{sampler_, src, VK_IMAGE_LAYOUT_GENERAL}, store{VK_NULL_HANDLE, dst, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &in;
        w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &store;
        push_(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 2, w);
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, constants);
        vkCmdDispatch(cmd, (out.width + 15) / 16, (out.height + 15) / 16, 1);  // 64 hilos, 16x16 píxeles cada grupo
    }

    const char* who_;
    bool failed_ = false, pending_layout_ = false;
    PFN_vkCmdPushDescriptorSetKHR push_ = nullptr;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    Pass easu_, rcas_, rcas_up_;
    Image mid_, out_;
    VkExtent2D extent_{};
    // blit_encoded: la copia UNORM de la textura de la ventana sRGB.
    Image raw_;
    VkExtent2D raw_extent_{};
    VkFormat raw_format_ = VK_FORMAT_UNDEFINED;
    bool raw_failed_ = false, raw_new_ = false;
};

Fsr g_fsr("fsr");
Fsr g_scene("fsr de la escena");  // el de la escena: otro tamaño de salida (ver record_scene)

}  // namespace

bool record(VkCommandBuffer cmd, VkImageView src, VkExtent2D src_extent, VkExtent2D viewport, VkImage dst,
            VkFormat dst_format, VkExtent2D dst_extent, bool easu) {
    return g_fsr.record(cmd, src, src_extent, viewport, dst, dst_format, dst_extent, easu);
}

bool record_scene(VkCommandBuffer cmd, VkImageView src, VkExtent2D src_extent, VkImageView dst, VkExtent2D dst_extent,
                  bool rcas) {
    return g_scene.scene(cmd, src, src_extent, dst, dst_extent, rcas);
}

bool blit_encoded(VkCommandBuffer cmd, VkImage src, VkFormat raw_format, VkExtent2D src_extent, VkRect2D rect, VkImage dst,
                  VkExtent2D dst_extent) {
    return g_fsr.blit_encoded(cmd, src, raw_format, src_extent, rect, dst, dst_extent);
}

}  // namespace nvn::fsr
