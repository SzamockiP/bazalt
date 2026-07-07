#pragma once
#include <volk.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <expected>
#include <array>
#include <map>
#include <unordered_map>
#include "ShaderCompiler.hpp"
#include "Renderer.hpp"

enum class Format {
    FLOAT2,
    FLOAT3,
    FLOAT4
};

enum class CullMode {
    NONE,
    BACK,
    FRONT,
    FRONT_AND_BACK
};

enum class FrontFace {
    CLOCKWISE,
    COUNTER_CLOCKWISE
};

class Pipeline {
public:
    // Maps binding index -> VkDescriptorType so DescriptorSet knows what type to write
    using BindingTypeMap = std::unordered_map<uint32_t, VkDescriptorType>;

    Pipeline(VkDevice device, VkPipeline pipeline, VkPipelineLayout layout, 
             std::vector<VkDescriptorSetLayout> descLayouts = {},
             std::map<uint32_t, BindingTypeMap> bindingTypes = {})
        : device_(device), pipeline_(pipeline), layout_(layout),
          desc_layouts_(std::move(descLayouts)),
          binding_types_(std::move(bindingTypes)) {}

    ~Pipeline() {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, layout_, nullptr);
        }
        for (auto dl : desc_layouts_) {
            if (dl != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, dl, nullptr);
            }
        }
    }

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Pipeline(Pipeline&& other) noexcept
        : device_(other.device_), pipeline_(other.pipeline_), layout_(other.layout_),
          desc_layouts_(std::move(other.desc_layouts_)),
          binding_types_(std::move(other.binding_types_)) {
        other.pipeline_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
    }

    Pipeline& operator=(Pipeline&& other) noexcept {
        if (this != &other) {
            if (pipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline_, nullptr);
            }
            if (layout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, layout_, nullptr);
            }
            for (auto dl : desc_layouts_) {
                if (dl != VK_NULL_HANDLE) {
                    vkDestroyDescriptorSetLayout(device_, dl, nullptr);
                }
            }
            device_ = other.device_;
            pipeline_ = other.pipeline_;
            layout_ = other.layout_;
            desc_layouts_ = std::move(other.desc_layouts_);
            binding_types_ = std::move(other.binding_types_);
            
            other.pipeline_ = VK_NULL_HANDLE;
            other.layout_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    VkPipeline get() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }

    VkDescriptorSetLayout descriptor_set_layout(uint32_t setIndex) const {
        if (setIndex < desc_layouts_.size()) {
            return desc_layouts_[setIndex];
        }
        return VK_NULL_HANDLE;
    }

    uint32_t descriptor_set_layout_count() const {
        return static_cast<uint32_t>(desc_layouts_.size());
    }

    const BindingTypeMap& binding_types(uint32_t setIndex) const {
        static const BindingTypeMap empty;
        auto it = binding_types_.find(setIndex);
        if (it != binding_types_.end()) {
            return it->second;
        }
        return empty;
    }

private:
    VkDevice device_;
    VkPipeline pipeline_;
    VkPipelineLayout layout_;
    std::vector<VkDescriptorSetLayout> desc_layouts_;
    std::map<uint32_t, BindingTypeMap> binding_types_;
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

    PipelineBuilder& cullMode(CullMode mode, FrontFace frontFace) {
        cull_mode_ = mode;
        front_face_ = frontFace;
        return *this;
    }

    PipelineBuilder& blend(bool enable) {
        blend_enable_ = enable;
        return *this;
    }

    PipelineBuilder& pushConstant(uint32_t size, ShaderStage stage) {
        VkPushConstantRange range{
            .stageFlags = static_cast<VkShaderStageFlags>((stage == ShaderStage::VERTEX) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT),
            .offset = 0,
            .size = size
        };
        push_constant_ranges_.push_back(range);
        return *this;
    }

    PipelineBuilder& uniformBuffer(uint32_t binding, ShaderStage stage, uint32_t set) {
        return addDescriptorBinding_(binding, stage, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, set);
    }

    PipelineBuilder& storageBuffer(uint32_t binding, ShaderStage stage, uint32_t set) {
        return addDescriptorBinding_(binding, stage, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, set);
    }

    PipelineBuilder& texture(uint32_t binding, ShaderStage stage, uint32_t set) {
        return addDescriptorBinding_(binding, stage, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, set);
    }

    std::expected<std::shared_ptr<Pipeline>, std::string> build() {
        if (!vertex_shader_ || !fragment_shader_) {
            return std::unexpected("Vertex and fragment shaders must be provided");
        }

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_->get(),
            .pName = "main",
            .pSpecializationInfo = nullptr
        };
        shaderStages.push_back(vertShaderStageInfo);

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_->get(),
            .pName = "main",
            .pSpecializationInfo = nullptr
        };
        shaderStages.push_back(fragShaderStageInfo);

        // Descriptor Set Layouts — one per set index
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        std::map<uint32_t, Pipeline::BindingTypeMap> allBindingTypes;

        if (!descriptor_bindings_.empty()) {
            // Find max set index
            uint32_t maxSetIndex = 0;
            for (const auto& [idx, _] : descriptor_bindings_) {
                if (idx > maxSetIndex) maxSetIndex = idx;
            }

            for (uint32_t s = 0; s <= maxSetIndex; s++) {
                auto it = descriptor_bindings_.find(s);
                if (it != descriptor_bindings_.end() && !it->second.empty()) {
                    VkDescriptorSetLayoutCreateInfo layoutInfo{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .bindingCount = static_cast<uint32_t>(it->second.size()),
                        .pBindings = it->second.data()
                    };

                    VkDescriptorSetLayout layout;
                    if (vkCreateDescriptorSetLayout(renderer_.device(), &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
                        // Cleanup already created layouts
                        for (auto dl : descriptorSetLayouts) {
                            vkDestroyDescriptorSetLayout(renderer_.device(), dl, nullptr);
                        }
                        return std::unexpected("Failed to create descriptor set layout for set " + std::to_string(s));
                    }
                    descriptorSetLayouts.push_back(layout);

                    // Build binding types map for this set
                    Pipeline::BindingTypeMap btm;
                    for (const auto& b : it->second) {
                        btm[b.binding] = b.descriptorType;
                    }
                    allBindingTypes[s] = std::move(btm);
                } else {
                    // Create empty layout for gap set indices
                    VkDescriptorSetLayoutCreateInfo layoutInfo{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .bindingCount = 0,
                        .pBindings = nullptr
                    };

                    VkDescriptorSetLayout layout;
                    if (vkCreateDescriptorSetLayout(renderer_.device(), &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
                        for (auto dl : descriptorSetLayouts) {
                            vkDestroyDescriptorSetLayout(renderer_.device(), dl, nullptr);
                        }
                        return std::unexpected("Failed to create empty descriptor set layout for set " + std::to_string(s));
                    }
                    descriptorSetLayouts.push_back(layout);
                }
            }
        }

        // Vertex Input
        VkVertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = 0, // Set later
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };

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

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE
        };

        // Dynamic State
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        VkPipelineViewportStateCreateInfo viewportState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr
        };

        VkCullModeFlags vkCullMode = VK_CULL_MODE_NONE;
        if (cull_mode_ == CullMode::BACK) vkCullMode = VK_CULL_MODE_BACK_BIT;
        else if (cull_mode_ == CullMode::FRONT) vkCullMode = VK_CULL_MODE_FRONT_BIT;
        else if (cull_mode_ == CullMode::FRONT_AND_BACK) vkCullMode = VK_CULL_MODE_FRONT_AND_BACK;

        VkFrontFace vkFrontFace = (front_face_ == FrontFace::CLOCKWISE) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineRasterizationStateCreateInfo rasterizer{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = vkCullMode,
            .frontFace = vkFrontFace,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f
        };

        VkPipelineMultisampleStateCreateInfo multisampling{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE
        };

        VkPipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = blend_enable_ ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = blend_enable_ ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = blend_enable_ ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = blend_enable_ ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = blend_enable_ ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
        };

        VkPipelineColorBlendStateCreateInfo colorBlending{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}
        };

        VkPipelineDepthStencilStateCreateInfo depthStencil{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthTestEnable = depth_test_ ? VK_TRUE : VK_FALSE,
            .depthWriteEnable = depth_test_ ? VK_TRUE : VK_FALSE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f
        };

        // Pipeline Layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
            .pSetLayouts = descriptorSetLayouts.empty() ? nullptr : descriptorSetLayouts.data(),
            .pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges_.size()),
            .pPushConstantRanges = push_constant_ranges_.data()
        };

        VkPipelineLayout pipelineLayout;
        if (vkCreatePipelineLayout(renderer_.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            for (auto dl : descriptorSetLayouts) {
                vkDestroyDescriptorSetLayout(renderer_.device(), dl, nullptr);
            }
            return std::unexpected("Failed to create pipeline layout!");
        }

        // Dynamic Rendering Info
        VkFormat colorFormat = renderer_.swapchain_format();
        VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = nullptr,
            .viewMask = 0,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &colorFormat,
            .depthAttachmentFormat = renderer_.depth_format() != VK_FORMAT_UNDEFINED ? renderer_.depth_format() : VK_FORMAT_UNDEFINED,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED
        };

        VkGraphicsPipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &pipelineRenderingCreateInfo,
            .flags = 0,
            .stageCount = static_cast<uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pTessellationState = nullptr,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            .renderPass = VK_NULL_HANDLE, // Dynamic rendering
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1
        };

        VkPipeline graphicsPipeline;
        if (vkCreateGraphicsPipelines(renderer_.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            vkDestroyPipelineLayout(renderer_.device(), pipelineLayout, nullptr);
            for (auto dl : descriptorSetLayouts) {
                vkDestroyDescriptorSetLayout(renderer_.device(), dl, nullptr);
            }
            return std::unexpected("Failed to create graphics pipeline!");
        }

        return std::make_shared<Pipeline>(renderer_.device(), graphicsPipeline, pipelineLayout,
                                          std::move(descriptorSetLayouts), std::move(allBindingTypes));
    }

private:
    PipelineBuilder& addDescriptorBinding_(uint32_t binding, ShaderStage stage, VkDescriptorType descriptorType, uint32_t setIndex) {
        VkShaderStageFlags stageFlag = (stage == ShaderStage::VERTEX) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        
        auto& bindings = descriptor_bindings_[setIndex];
        
        for (auto& b : bindings) {
            if (b.binding == binding) {
                b.stageFlags |= stageFlag;
                return *this;
            }
        }

        VkDescriptorSetLayoutBinding layoutBinding{
            .binding = binding,
            .descriptorType = descriptorType,
            .descriptorCount = 1,
            .stageFlags = stageFlag,
            .pImmutableSamplers = nullptr
        };
        bindings.push_back(layoutBinding);
        return *this;
    }

    Renderer& renderer_;
    std::shared_ptr<ShaderModule> vertex_shader_;
    std::shared_ptr<ShaderModule> fragment_shader_;
    std::vector<Format> formats_;
    bool depth_test_ = false;
    CullMode cull_mode_ = CullMode::BACK;
    FrontFace front_face_ = FrontFace::COUNTER_CLOCKWISE;
    bool blend_enable_ = false;
    std::vector<VkPushConstantRange> push_constant_ranges_;
    std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> descriptor_bindings_;
};
