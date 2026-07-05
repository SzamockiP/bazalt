#pragma once
#include <volk.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <expected>
#include <array>
#include <unordered_map>
#include "ShaderCompiler.hpp"
#include "Renderer.hpp"

enum class Format {
    FLOAT2,
    FLOAT3,
    FLOAT4
};

class Pipeline {
public:
    // Maps binding index -> VkDescriptorType so CommandBuffer knows what type to write
    using BindingTypeMap = std::unordered_map<uint32_t, VkDescriptorType>;

    Pipeline(VkDevice device, VkPipeline pipeline, VkPipelineLayout layout, 
             VkDescriptorSetLayout descLayout = VK_NULL_HANDLE, 
             VkDescriptorPool descPool = VK_NULL_HANDLE, 
             std::array<VkDescriptorSet, Renderer::MAX_FRAMES_IN_FLIGHT> descSets = {VK_NULL_HANDLE},
             BindingTypeMap bindingTypes = {})
        : device_(device), pipeline_(pipeline), layout_(layout),
          desc_layout_(descLayout), desc_pool_(descPool), desc_sets_(descSets),
          binding_types_(std::move(bindingTypes)) {}

    ~Pipeline() {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, layout_, nullptr);
        }
        if (desc_pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
        }
        if (desc_layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr);
        }
    }

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Pipeline(Pipeline&& other) noexcept
        : device_(other.device_), pipeline_(other.pipeline_), layout_(other.layout_),
          desc_layout_(other.desc_layout_), desc_pool_(other.desc_pool_), desc_sets_(other.desc_sets_),
          bound_buffers_(other.bound_buffers_), binding_types_(std::move(other.binding_types_)) {
        other.pipeline_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
        other.desc_layout_ = VK_NULL_HANDLE;
        other.desc_pool_ = VK_NULL_HANDLE;
        other.desc_sets_.fill(VK_NULL_HANDLE);
    }

    Pipeline& operator=(Pipeline&& other) noexcept {
        if (this != &other) {
            if (pipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline_, nullptr);
            }
            if (layout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, layout_, nullptr);
            }
            if (desc_pool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
            }
            if (desc_layout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr);
            }
            device_ = other.device_;
            pipeline_ = other.pipeline_;
            layout_ = other.layout_;
            desc_layout_ = other.desc_layout_;
            desc_pool_ = other.desc_pool_;
            desc_sets_ = other.desc_sets_;
            bound_buffers_ = other.bound_buffers_;
            binding_types_ = std::move(other.binding_types_);
            
            other.pipeline_ = VK_NULL_HANDLE;
            other.layout_ = VK_NULL_HANDLE;
            other.desc_layout_ = VK_NULL_HANDLE;
            other.desc_pool_ = VK_NULL_HANDLE;
            other.desc_sets_.fill(VK_NULL_HANDLE);
        }
        return *this;
    }

    VkPipeline get() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSet descriptor_set(uint32_t frame) const { return desc_sets_[frame]; }

    VkBuffer get_bound_buffer(uint32_t frame) const { return bound_buffers_[frame]; }
    void set_bound_buffer(uint32_t frame, VkBuffer buffer) { bound_buffers_[frame] = buffer; }

    VkDescriptorType descriptor_type_for_binding(uint32_t binding) const {
        auto it = binding_types_.find(binding);
        if (it != binding_types_.end()) {
            return it->second;
        }
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // default fallback
    }

private:
    VkDevice device_;
    VkPipeline pipeline_;
    VkPipelineLayout layout_;
    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, Renderer::MAX_FRAMES_IN_FLIGHT> desc_sets_ = {VK_NULL_HANDLE};
    std::array<VkBuffer, Renderer::MAX_FRAMES_IN_FLIGHT> bound_buffers_ = {VK_NULL_HANDLE};
    BindingTypeMap binding_types_;
};

class PipelineBuilder {
public:
    PipelineBuilder(Renderer& renderer) : renderer_(renderer) {}

    PipelineBuilder& vertexShader(std::shared_ptr<ShaderModule> shader) {
        vertex_shader_ = shader;
        return *this;
    }

    PipelineBuilder& fragmentShader(std::shared_ptr<ShaderModule> shader) {
        fragment_shader_ = shader;
        return *this;
    }

    PipelineBuilder& vertexFormat(const std::vector<Format>& formats) {
        formats_ = formats;
        return *this;
    }

    PipelineBuilder& depthTest(bool enable) {
        depth_test_ = enable;
        return *this;
    }

    PipelineBuilder& pushConstant(uint32_t size, ShaderStage stage) {
        VkPushConstantRange range{};
        range.stageFlags = (stage == ShaderStage::VERTEX) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = size;
        push_constant_ranges_.push_back(range);
        return *this;
    }

    PipelineBuilder& uniformBuffer(uint32_t binding, ShaderStage stage) {
        return descriptorBuffer_(binding, stage, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    }

    PipelineBuilder& storageBuffer(uint32_t binding, ShaderStage stage) {
        return descriptorBuffer_(binding, stage, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }

    std::expected<std::shared_ptr<Pipeline>, std::string> build() {
        if (!vertex_shader_ || !fragment_shader_) {
            return std::unexpected("Vertex and fragment shaders must be provided");
        }

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertex_shader_->get();
        vertShaderStageInfo.pName = "main";
        shaderStages.push_back(vertShaderStageInfo);

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragment_shader_->get();
        fragShaderStageInfo.pName = "main";
        shaderStages.push_back(fragShaderStageInfo);

        // Descriptor Sets
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, Renderer::MAX_FRAMES_IN_FLIGHT> descriptorSets = {VK_NULL_HANDLE};
        Pipeline::BindingTypeMap bindingTypes;

        if (!descriptor_bindings_.empty()) {
            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(descriptor_bindings_.size());
            layoutInfo.pBindings = descriptor_bindings_.data();

            if (vkCreateDescriptorSetLayout(renderer_.device(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
                return std::unexpected("Failed to create descriptor set layout!");
            }

            // Build binding type map and count descriptors per type for pool sizes
            std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
            for (const auto& b : descriptor_bindings_) {
                typeCounts[b.descriptorType] += Renderer::MAX_FRAMES_IN_FLIGHT;
                bindingTypes[b.binding] = b.descriptorType;
            }

            std::vector<VkDescriptorPoolSize> poolSizes;
            for (const auto& [type, count] : typeCounts) {
                VkDescriptorPoolSize ps{};
                ps.type = type;
                ps.descriptorCount = count;
                poolSizes.push_back(ps);
            }

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            poolInfo.pPoolSizes = poolSizes.data();
            poolInfo.maxSets = Renderer::MAX_FRAMES_IN_FLIGHT;

            if (vkCreateDescriptorPool(renderer_.device(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
                vkDestroyDescriptorSetLayout(renderer_.device(), descriptorSetLayout, nullptr);
                return std::unexpected("Failed to create descriptor pool!");
            }

            std::vector<VkDescriptorSetLayout> layouts(Renderer::MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptorPool;
            allocInfo.descriptorSetCount = Renderer::MAX_FRAMES_IN_FLIGHT;
            allocInfo.pSetLayouts = layouts.data();

            if (vkAllocateDescriptorSets(renderer_.device(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
                vkDestroyDescriptorPool(renderer_.device(), descriptorPool, nullptr);
                vkDestroyDescriptorSetLayout(renderer_.device(), descriptorSetLayout, nullptr);
                return std::unexpected("Failed to allocate descriptor set!");
            }
        }

        // Vertex Input
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::vector<VkVertexInputAttributeDescription> attributeDescriptions(formats_.size());
        uint32_t offset = 0;
        for (size_t i = 0; i < formats_.size(); i++) {
            attributeDescriptions[i].binding = 0;
            attributeDescriptions[i].location = static_cast<uint32_t>(i);
            
            switch (formats_[i]) {
                case Format::FLOAT2: 
                    attributeDescriptions[i].format = VK_FORMAT_R32G32_SFLOAT;
                    attributeDescriptions[i].offset = offset;
                    offset += 8;
                    break;
                case Format::FLOAT3:
                    attributeDescriptions[i].format = VK_FORMAT_R32G32B32_SFLOAT;
                    attributeDescriptions[i].offset = offset;
                    offset += 12;
                    break;
                case Format::FLOAT4:
                    attributeDescriptions[i].format = VK_FORMAT_R32G32B32A32_SFLOAT;
                    attributeDescriptions[i].offset = offset;
                    offset += 16;
                    break;
            }
        }
        bindingDescription.stride = offset;

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        if (!formats_.empty()) {
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        }

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // Dynamic State
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = depth_test_ ? VK_TRUE : VK_FALSE;
        depthStencil.depthWriteEnable = depth_test_ ? VK_TRUE : VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        // Pipeline Layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges_.size());
        pipelineLayoutInfo.pPushConstantRanges = push_constant_ranges_.data();
        
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        }

        VkPipelineLayout pipelineLayout;
        if (vkCreatePipelineLayout(renderer_.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            if (descriptorPool) vkDestroyDescriptorPool(renderer_.device(), descriptorPool, nullptr);
            if (descriptorSetLayout) vkDestroyDescriptorSetLayout(renderer_.device(), descriptorSetLayout, nullptr);
            return std::unexpected("Failed to create pipeline layout!");
        }

        // Dynamic Rendering Info
        VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
        pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        VkFormat colorFormat = renderer_.swapchain_format();
        pipelineRenderingCreateInfo.colorAttachmentCount = 1;
        pipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
        if (renderer_.depth_format() != VK_FORMAT_UNDEFINED) {
            pipelineRenderingCreateInfo.depthAttachmentFormat = renderer_.depth_format();
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = &pipelineRenderingCreateInfo;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = VK_NULL_HANDLE; // Dynamic rendering
        pipelineInfo.subpass = 0;

        VkPipeline graphicsPipeline;
        if (vkCreateGraphicsPipelines(renderer_.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            vkDestroyPipelineLayout(renderer_.device(), pipelineLayout, nullptr);
            if (descriptorPool) vkDestroyDescriptorPool(renderer_.device(), descriptorPool, nullptr);
            if (descriptorSetLayout) vkDestroyDescriptorSetLayout(renderer_.device(), descriptorSetLayout, nullptr);
            return std::unexpected("Failed to create graphics pipeline!");
        }

        return std::make_shared<Pipeline>(renderer_.device(), graphicsPipeline, pipelineLayout, descriptorSetLayout, descriptorPool, descriptorSets, std::move(bindingTypes));
    }

private:
    PipelineBuilder& descriptorBuffer_(uint32_t binding, ShaderStage stage, VkDescriptorType descriptorType) {
        VkShaderStageFlags stageFlag = (stage == ShaderStage::VERTEX) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        
        for (auto& b : descriptor_bindings_) {
            if (b.binding == binding) {
                b.stageFlags |= stageFlag;
                return *this;
            }
        }

        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = descriptorType;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = stageFlag;
        layoutBinding.pImmutableSamplers = nullptr;
        descriptor_bindings_.push_back(layoutBinding);
        return *this;
    }

    Renderer& renderer_;
    std::shared_ptr<ShaderModule> vertex_shader_;
    std::shared_ptr<ShaderModule> fragment_shader_;
    std::vector<Format> formats_;
    bool depth_test_ = false;
    std::vector<VkPushConstantRange> push_constant_ranges_;
    std::vector<VkDescriptorSetLayoutBinding> descriptor_bindings_;
};
