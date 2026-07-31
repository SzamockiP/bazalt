#include "Bindings.hpp"

void bind_pipelines(py::module_& m)
{
    py::class_<Pipeline, std::shared_ptr<Pipeline>>(m, "Pipeline");

    // Lambdas, not member pointers: the setters take a deducing-this object
    // parameter, so &GraphicsPipelineBuilder::vertex_shader would be a plain
    // function pointer that .def() cannot treat as a method.
    py::class_<GraphicsPipelineBuilder, std::shared_ptr<GraphicsPipelineBuilder>>(m, "GraphicsPipelineBuilder")
        .def(
            "vertex_shader",
            [](GraphicsPipelineBuilder& self, std::shared_ptr<ShaderModule> shader) -> GraphicsPipelineBuilder&
            { return self.vertex_shader(std::move(shader)); },
            py::arg("shader"))
        .def(
            "fragment_shader",
            [](GraphicsPipelineBuilder& self, std::shared_ptr<ShaderModule> shader) -> GraphicsPipelineBuilder&
            { return self.fragment_shader(std::move(shader)); },
            py::arg("shader"))
        .def(
            "tess_control_shader",
            [](GraphicsPipelineBuilder& self, std::shared_ptr<ShaderModule> shader) -> GraphicsPipelineBuilder&
            { return self.tess_control_shader(std::move(shader)); },
            py::arg("shader"))
        .def(
            "tess_evaluation_shader",
            [](GraphicsPipelineBuilder& self, std::shared_ptr<ShaderModule> shader) -> GraphicsPipelineBuilder&
            { return self.tess_evaluation_shader(std::move(shader)); },
            py::arg("shader"))
        .def(
            "geometry_shader",
            [](GraphicsPipelineBuilder& self, std::shared_ptr<ShaderModule> shader) -> GraphicsPipelineBuilder&
            { return self.geometry_shader(std::move(shader)); },
            py::arg("shader"))
        .def(
            "patch_control_points",
            [](GraphicsPipelineBuilder& self, std::uint32_t count) -> GraphicsPipelineBuilder&
            { return self.patch_control_points(count); },
            py::arg("count"))
        .def(
            "vertex_format",
            [](GraphicsPipelineBuilder& self, const std::vector<VertexFormat>& formats) -> GraphicsPipelineBuilder&
            { return self.vertex_format(formats); },
            py::arg("formats"))
        .def(
            "instance_format",
            [](GraphicsPipelineBuilder& self, const std::vector<VertexFormat>& formats) -> GraphicsPipelineBuilder&
            { return self.instance_format(formats); },
            py::arg("formats"))
        .def(
            "depth_test",
            [](GraphicsPipelineBuilder& self, bool enable, bool write, CompareOp compare) -> GraphicsPipelineBuilder&
            { return self.depth_test(enable, write, compare); },
            py::arg("enable"),
            py::arg("write") = true,
            py::arg("compare") = CompareOp::LESS_OR_EQUAL)
        .def(
            "cull_mode",
            [](GraphicsPipelineBuilder& self, CullMode mode, FrontFace front_face) -> GraphicsPipelineBuilder&
            { return self.cull_mode(mode, front_face); },
            py::arg("mode"),
            py::arg("front_face"))
        .def(
            "polygon_mode",
            [](GraphicsPipelineBuilder& self, PolygonMode mode) -> GraphicsPipelineBuilder&
            { return self.polygon_mode(mode); },
            py::arg("mode"))
        .def(
            "line_width",
            [](GraphicsPipelineBuilder& self, float width) -> GraphicsPipelineBuilder&
            { return self.line_width(width); },
            py::arg("width"))
        .def(
            "depth_bias",
            [](GraphicsPipelineBuilder& self, float constant, float slope) -> GraphicsPipelineBuilder&
            { return self.depth_bias(constant, slope); },
            py::arg("constant"),
            py::arg("slope") = 0.0f)
        .def(
            "blend",
            [](GraphicsPipelineBuilder& self, bool enable, BlendMode mode, std::optional<std::uint32_t> attachment)
                -> GraphicsPipelineBuilder&
            { return self.blend(enable, mode, attachment ? static_cast<int>(*attachment) : -1); },
            py::arg("enable"),
            py::arg("mode") = BlendMode::ALPHA,
            py::arg("attachment") = py::none())
        .def(
            "color_mask",
            [](GraphicsPipelineBuilder& self,
               bool red,
               bool green,
               bool blue,
               bool alpha,
               std::optional<std::uint32_t> attachment) -> GraphicsPipelineBuilder&
            { return self.color_mask(red, green, blue, alpha, attachment ? static_cast<int>(*attachment) : -1); },
            py::arg("red") = true,
            py::arg("green") = true,
            py::arg("blue") = true,
            py::arg("alpha") = true,
            py::arg("attachment") = py::none())
        .def(
            "stencil_test",
            [](GraphicsPipelineBuilder& self,
               bool enable,
               CompareOp compare,
               std::uint32_t ref,
               StencilOp pass_op,
               StencilOp fail_op,
               StencilOp depth_fail_op,
               std::uint32_t read_mask,
               std::uint32_t write_mask) -> GraphicsPipelineBuilder&
            { return self.stencil_test(enable, compare, ref, pass_op, fail_op, depth_fail_op, read_mask, write_mask); },
            py::arg("enable"),
            py::arg("compare") = CompareOp::ALWAYS,
            py::arg("ref") = 0,
            py::arg("pass_op") = StencilOp::KEEP,
            py::arg("fail_op") = StencilOp::KEEP,
            py::arg("depth_fail_op") = StencilOp::KEEP,
            py::arg("read_mask") = 0xFFu,
            py::arg("write_mask") = 0xFFu)
        .def(
            "depth_clamp",
            [](GraphicsPipelineBuilder& self, bool enable) -> GraphicsPipelineBuilder&
            { return self.depth_clamp(enable); },
            py::arg("enable") = true)
        .def(
            "alpha_to_coverage",
            [](GraphicsPipelineBuilder& self, bool enable) -> GraphicsPipelineBuilder&
            { return self.alpha_to_coverage(enable); },
            py::arg("enable") = true)
        // A bool IS an int in Python, so it has to be tested first or True would
        // be baked in as the integer 1 and a `bool` constant in the shader would
        // read whatever that bit pattern means.
        .def(
            "constant",
            [](GraphicsPipelineBuilder& self, std::uint32_t id, const py::object& value, ShaderStage stage)
                -> GraphicsPipelineBuilder& { return self.constant(id, spec_constant_bytes(value), stage); },
            py::arg("id"),
            py::arg("value"),
            py::arg("stage"))
        .def(
            "topology",
            [](GraphicsPipelineBuilder& self, Topology topology) -> GraphicsPipelineBuilder&
            { return self.topology(topology); },
            py::arg("topology"))
        .def(
            "sample_shading",
            [](GraphicsPipelineBuilder& self, bool enable, float min_fraction) -> GraphicsPipelineBuilder&
            { return self.sample_shading(enable, min_fraction); },
            py::arg("enable") = true,
            py::arg("min_fraction") = 1.0f)
        .def(
            "push_constant",
            [](GraphicsPipelineBuilder& self, uint32_t size, ShaderStage stage) -> GraphicsPipelineBuilder&
            { return self.push_constant(size, stage); },
            py::arg("size"),
            py::arg("stage"))
        .def(
            "uniform_buffer",
            [](GraphicsPipelineBuilder& self,
               uint32_t binding,
               ShaderStage stage,
               uint32_t set,
               uint32_t count,
               std::optional<bool> update_after_bind) -> GraphicsPipelineBuilder&
            { return self.uniform_buffer(binding, stage, set, count, update_after_bind); },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set"),
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none())
        .def(
            "storage_buffer",
            [](GraphicsPipelineBuilder& self,
               uint32_t binding,
               ShaderStage stage,
               uint32_t set,
               uint32_t count,
               std::optional<bool> update_after_bind) -> GraphicsPipelineBuilder&
            { return self.storage_buffer(binding, stage, set, count, update_after_bind); },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set"),
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none())
        // count>1 declares a descriptor array: one binding holding N textures,
        // written with set_image(..., index=i) and indexed in the shader.
        .def(
            "texture",
            [](GraphicsPipelineBuilder& self,
               uint32_t binding,
               ShaderStage stage,
               uint32_t set,
               uint32_t count,
               std::optional<bool> update_after_bind) -> GraphicsPipelineBuilder&
            { return self.texture(binding, stage, set, count, update_after_bind); },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set"),
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none())
        .def(
            "storage_image",
            [](GraphicsPipelineBuilder& self,
               uint32_t binding,
               ShaderStage stage,
               uint32_t set,
               uint32_t count,
               std::optional<bool> update_after_bind) -> GraphicsPipelineBuilder&
            { return self.storage_image(binding, stage, set, count, update_after_bind); },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set"),
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none())
        // Takes any RenderTarget. A SwapchainRenderer *is* one, so windowed code
        // reads the same as offscreen code — build(renderer) still works, it just
        // isn't a special case any more.
        .def(
            "name",
            [](GraphicsPipelineBuilder& self, std::string name) -> GraphicsPipelineBuilder&
            { return self.name(std::move(name)); },
            py::arg("name"))
        .def(
            "build",
            [](GraphicsPipelineBuilder& builder, std::shared_ptr<RenderTarget> target) -> py::object
            {
                require_same_context(&builder.context(), target->owner(), "build");
                auto pipeline = unwrap(builder.build(*target), nullptr);
                // Watch unconditionally: a pipeline whose shaders were all unwatched
                // (source=, .spv from a gone file) simply never fires.
                if (auto* hr = builder.context().hot_reload())
                    hr->watch_pipeline(pipeline);
                return py::cast(pipeline);
            },
            py::arg("target"));

    // No stage arguments anywhere: compute has exactly one stage, so asking for
    // it could only ever be redundant or wrong. build() takes no target —
    // compute has no attachments.
    py::class_<ComputePipelineBuilder, std::shared_ptr<ComputePipelineBuilder>>(m, "ComputePipelineBuilder")
        .def(
            "shader",
            [](ComputePipelineBuilder& self, std::shared_ptr<ShaderModule> shader) -> ComputePipelineBuilder&
            { return self.shader(std::move(shader)); },
            py::arg("shader"))
        .def(
            "uniform_buffer",
            [](ComputePipelineBuilder& self,
               uint32_t binding,
               uint32_t set,
               uint32_t count,
               std::optional<bool> update_after_bind) -> ComputePipelineBuilder&
            { return self.uniform_buffer(binding, set, count, update_after_bind); },
            py::arg("binding"),
            py::arg("set") = 0,
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none())
        .def(
            "storage_buffer",
            [](ComputePipelineBuilder& self,
               uint32_t binding,
               uint32_t set,
               uint32_t count,
               std::optional<bool> update_after_bind) -> ComputePipelineBuilder&
            { return self.storage_buffer(binding, set, count, update_after_bind); },
            py::arg("binding"),
            py::arg("set") = 0,
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none())
        // A sampled image in a compute shader: filtering, mips and address modes,
        // which a storage image has none of. The declarator was simply missing
        // until 0.21 -- everything downstream already handled it.
        .def(
            "texture",
            [](ComputePipelineBuilder& self,
               uint32_t binding,
               uint32_t set,
               uint32_t count,
               std::optional<bool> update_after_bind) -> ComputePipelineBuilder&
            { return self.texture(binding, set, count, update_after_bind); },
            py::arg("binding"),
            py::arg("set") = 0,
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none())
        .def(
            "storage_image",
            [](ComputePipelineBuilder& self,
               uint32_t binding,
               uint32_t set,
               uint32_t count,
               std::optional<bool> update_after_bind) -> ComputePipelineBuilder&
            { return self.storage_image(binding, set, count, update_after_bind); },
            py::arg("binding"),
            py::arg("set") = 0,
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none())
        .def(
            "push_constant",
            [](ComputePipelineBuilder& self, uint32_t size) -> ComputePipelineBuilder&
            { return self.push_constant(size); },
            py::arg("size"))
        .def(
            "constant",
            [](ComputePipelineBuilder& self, std::uint32_t id, const py::object& value) -> ComputePipelineBuilder&
            { return self.constant(id, spec_constant_bytes(value)); },
            py::arg("id"),
            py::arg("value"))
        .def(
            "name",
            [](ComputePipelineBuilder& self, std::string name) -> ComputePipelineBuilder&
            { return self.name(std::move(name)); },
            py::arg("name"))
        .def(
            "build",
            [](ComputePipelineBuilder& builder) -> py::object
            {
                auto pipeline = unwrap(builder.build(), nullptr);
                if (auto* hr = builder.context().hot_reload())
                    hr->watch_pipeline(pipeline);
                return py::cast(pipeline);
            });

    py::class_<DescriptorSet, std::shared_ptr<DescriptorSet>>(m, "DescriptorSet")
        // index= picks the element of a count>1 array binding. Writing the same
        // (binding, index) again replaces what was there.
        .def(
            "set_image",
            [](DescriptorSet& self,
               uint32_t binding,
               std::shared_ptr<Image> image,
               std::shared_ptr<Sampler> sampler,
               uint32_t index)
            {
                require_same_context(self.owner(), image->owner(), "set_image");
                unwrap(self.set_image(binding, std::move(image), std::move(sampler), index), nullptr);
            },
            py::arg("binding"),
            py::arg("image"),
            py::arg("sampler") = py::none(),
            // Keyword-only: set_image(0, img, 3) read as "index 3" and passed 3 as
            // a sampler. Everywhere else in the API the extras are keyword-only,
            // and the sibling verbs follow so the rule stays one rule (0.23).
            py::kw_only(),
            py::arg("index") = 0)
        .def(
            "set_storage_image",
            [](DescriptorSet& self, uint32_t binding, std::shared_ptr<Image> image, uint32_t index)
            {
                require_same_context(self.owner(), image->owner(), "set_storage_image");
                unwrap(self.set_storage_image(binding, std::move(image), index), nullptr);
            },
            py::arg("binding"),
            py::arg("image"),
            py::kw_only(),
            py::arg("index") = 0)
        .def(
            "set_buffer",
            [](DescriptorSet& self, uint32_t binding, std::shared_ptr<Buffer> buffer, uint32_t index)
            {
                require_same_context(self.owner(), buffer->owner(), "set_buffer");
                unwrap(self.set_buffer(binding, std::move(buffer), index), nullptr);
            },
            py::arg("binding"),
            py::arg("buffer"),
            py::kw_only(),
            py::arg("index") = 0);

    py::class_<DescriptorPool, std::shared_ptr<DescriptorPool>>(m, "DescriptorPool")
        .def(
            "allocate_set",
            [](DescriptorPool& pool, std::shared_ptr<Pipeline> pipeline, uint32_t setIndex) -> py::object
            {
                require_same_context(pool.owner(), pipeline->owner(), "allocate_set");
                return py::cast(unwrap(pool.allocate_descriptor_set(pipeline, setIndex), pool.logger().get()));
            },
            py::arg("pipeline"),
            py::arg("set"))
        .def(
            "allocate_frame_set",
            [](DescriptorPool& pool, std::shared_ptr<Pipeline> pipeline, uint32_t setIndex) -> py::object
            {
                require_same_context(pool.owner(), pipeline->owner(), "allocate_frame_set");
                return py::cast(unwrap(pool.allocate_frame_descriptor_set(pipeline, setIndex), pool.logger().get()));
            },
            py::arg("pipeline"),
            py::arg("set"));
}
