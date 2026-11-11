// FSR 1 al presentar (ver fsr.h). Todo en el hilo que presenta, con executor_mutex() tomado por quien llama.

#include "nnsdk/nvn/fsr.h"

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
    bool record(VkCommandBuffer cmd, VkImageView src, VkExtent2D in, VkExtent2D viewport, VkImage dst, VkFormat format,
                VkExtent2D out) {
        if (failed_ || !src || format != VK_FORMAT_R8G8B8A8_UNORM || NVN_FLAG("sin_fsr")) return false;
        if (!setup() || !images(out)) return false;

        // Lo de antes en la cola (el frame del juego, el FSR del frame anterior que leía estas imágenes) antes que esto.
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);

        // EASU: de src (in) a mid_ (out). Las constantes, como FsrEasuCon de ffx_fsr1.h.
        // con0 va con la zona que se escala (viewport) y el resto con el tamaño de la textura, como en FsrEasuCon.
        const float iw = static_cast<float>(in.width), ih = static_cast<float>(in.height);
        const float vw = static_cast<float>(viewport.width), vh = static_cast<float>(viewport.height);
        const float ow = static_cast<float>(out.width), oh = static_cast<float>(out.height);
        const uint32_t easu[16] = {as_uint(vw / ow), as_uint(vh / oh), as_uint(0.5f * vw / ow - 0.5f), as_uint(0.5f * vh / oh - 0.5f),
                                   as_uint(1.0f / iw), as_uint(1.0f / ih), as_uint(1.0f / iw), as_uint(-1.0f / ih),
                                   as_uint(-1.0f / iw), as_uint(2.0f / ih), as_uint(1.0f / iw), as_uint(2.0f / ih),
                                   as_uint(0.0f), as_uint(4.0f / ih), 0, 0};
        dispatch(cmd, easu_, src, mid_.view, easu, sizeof easu, out);
        VkMemoryBarrier rw{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        rw.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rw.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &rw, 0,
                             nullptr, 0, nullptr);

        // RCAS: de mid_ a out_, con la nitidez en "stops" (0 = la máxima; 0,25 un poco menos, sin halos).
        const uint32_t rcas[4] = {as_uint(std::exp2(-0.25f)), 0, 0, 0};
        dispatch(cmd, rcas_, mid_.view, out_.view, rcas, sizeof rcas, out);
        rw.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &rw, 0, nullptr,
                             0, nullptr);

        VkImageCopy region{};
        region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {out.width, out.height, 1};
        vkCmdCopyImage(cmd, out_.image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        return true;
    }

private:
    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    struct Pass {
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    bool fail(const char* why) {
        NVN_LOG("fsr: %s; se escala con el blit de siempre\n", why);
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
        NVN_LOG("fsr: listo (EASU + RCAS, %s)\n", half ? "FP16" : "FP32");
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

    bool image(Image& img, VkExtent2D extent, VkImageUsageFlags usage) {
        VkContext* c = vk();
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R8G8B8A8_UNORM;
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
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
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
            VkImageMemoryBarrier b[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
            for (int k = 0; k < 2; k++) {
                b[k].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                b[k].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b[k].srcQueueFamilyIndex = b[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b[k].image = k ? out_.image : mid_.image;
                b[k].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            }
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                                 nullptr, 2, b);
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

    bool failed_ = false, pending_layout_ = false;
    PFN_vkCmdPushDescriptorSetKHR push_ = nullptr;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    Pass easu_, rcas_;
    Image mid_, out_;
    VkExtent2D extent_{};
};

Fsr g_fsr;

}  // namespace

bool record(VkCommandBuffer cmd, VkImageView src, VkExtent2D src_extent, VkExtent2D viewport, VkImage dst,
            VkFormat dst_format, VkExtent2D dst_extent) {
    return g_fsr.record(cmd, src, src_extent, viewport, dst, dst_format, dst_extent);
}

}  // namespace nvn::fsr
