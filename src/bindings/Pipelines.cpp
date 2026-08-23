#include "Bindings.hpp"

void bind_pipelines(py::module_& m)
{
    // Registered and nothing more: a Pipeline is built by a builder and handed
    // to cmd.bind_pipeline, so it has no methods of its own. Named rather than a
    // discarded temporary — see the comment in Targets.cpp.
    [[maybe_unused]] const py::class_<Pipeline, std::shared_ptr<Pipeline>> pipeline_type(m, "Pipeline");

    py::class_<GraphicsPipelineBuilder, std::shared_ptr<GraphicsPipelineBuilder>> graphics(
        m, "GraphicsPipelineBuilder");

    using GraphicsShaderSetter = GraphicsPipelineBuilder& (GraphicsPipelineBuilder::*)(std::shared_ptr<ShaderModule>);
    for (auto [name, setter] : std::initializer_list<std::pair<const char*, GraphicsShaderSetter>>{
             {"vertex_shader", &GraphicsPipelineBuilder::vertex_shader},
             {"fragment_shader", &GraphicsPipelineBuilder::fragment_shader},
             {"tess_control_shader", &GraphicsPipelineBuilder::tess_control_shader},
             {"tess_evaluation_shader", &GraphicsPipelineBuilder::tess_evaluation_shader},
             {"geometry_shader", &GraphicsPipelineBuilder::geometry_shader}})
    {
        graphics.def(name, setter, py::arg("shader"));
    }

    graphics
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
            py::arg("front_face") = FrontFace::COUNTER_CLOCKWISE)
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
            [](GraphicsPipelineBuilder& self,
               bool enable,
               std::optional<BlendMode> mode,
               std::optional<BlendFactor> src,
               std::optional<BlendFactor> dst,
               std::optional<BlendOp> op,
               std::optional<BlendFactor> src_alpha,
               std::optional<BlendFactor> dst_alpha,
               std::optional<BlendOp> alpha_op,
               std::optional<std::uint32_t> attachment) -> GraphicsPipelineBuilder&
            {
                return self.blend(
                    enable,
                    resolve_blend_equation(mode, src, dst, op, src_alpha, dst_alpha, alpha_op),
                    attachment ? static_cast<int>(*attachment) : -1);
            },
            py::arg("enable"),
            py::arg("mode") = py::none(),
            // Everything past the mode is keyword-only. blend(True, MULTIPLY, 1)
            // reading as "attachment 1" is the trap set_image's index= had, and
            // the factors would make it worse (0.23).
            py::kw_only(),
            py::arg("src") = py::none(),
            py::arg("dst") = py::none(),
            py::arg("op") = py::none(),
            py::arg("src_alpha") = py::none(),
            py::arg("dst_alpha") = py::none(),
            py::arg("alpha_op") = py::none(),
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
               std::uint32_t write_mask,
               Face face) -> GraphicsPipelineBuilder&
            {
                return self.stencil_test(
                    enable, compare, ref, pass_op, fail_op, depth_fail_op, read_mask, write_mask, face);
            },
            py::arg("enable"),
            py::arg("compare") = CompareOp::ALWAYS,
            py::arg("ref") = 0,
            py::arg("pass_op") = StencilOp::KEEP,
            py::arg("fail_op") = StencilOp::KEEP,
            py::arg("depth_fail_op") = StencilOp::KEEP,
            py::arg("read_mask") = 0xFFu,
            py::arg("write_mask") = 0xFFu,
            py::arg("face") = Face::FRONT_AND_BACK)
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
            [](GraphicsPipelineBuilder& self, Topology topology, bool restart) -> GraphicsPipelineBuilder&
            { return self.topology(topology, restart); },
            py::arg("topology"),
            py::kw_only(),
            py::arg("restart") = false)
        .def(
            "sample_shading",
            [](GraphicsPipelineBuilder& self, bool enable, float min_fraction) -> GraphicsPipelineBuilder&
            { return self.sample_shading(enable, min_fraction); },
            py::arg("enable") = true,
            py::arg("min_fraction") = 1.0f)
        .def(
            "push_constant",
            [](GraphicsPipelineBuilder& self, uint32_t size, const py::object& stage) -> GraphicsPipelineBuilder&
            {
                // One VkPushConstantRange per stage at offset 0 — the same
                // bytes both stages read, which is what users write today as
                // two calls.
                for (const ShaderStage s : stages_of(stage, "push_constant"))
                {
                    self.push_constant(size, s);
                }
                return self;
            },
            py::arg("size"),
            py::arg("stage"))
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
            [](GraphicsPipelineBuilder& builder, const std::shared_ptr<RenderTarget>& target) -> py::object
            {
                require_same_context(&builder.context(), target->owner(), "build");
                auto pipeline = unwrap(builder.build(*target), nullptr);
                // Watch unconditionally: a pipeline whose shaders were all unwatched
                // (source=, .spv from a gone file) simply never fires.
                if (auto* hr = builder.context().hot_reload())
                {
                    hr->watch_pipeline(pipeline);
                }
                return py::cast(pipeline);
            },
            py::arg("target"));

    // The four descriptor declarators share one signature and one arg list.
    // set= defaulted to match the compute builder: the same declarator asked
    // for the set on one side and assumed it on the other, so a single-set
    // pipeline paid `set=0` on every line for nothing.
    using GraphicsDeclarator = GraphicsPipelineBuilder& (
        GraphicsPipelineBuilder::*)(uint32_t, ShaderStage, uint32_t, uint32_t, std::optional<bool>);
    for (auto [name, declarator] : std::initializer_list<std::pair<const char*, GraphicsDeclarator>>{
             {"uniform_buffer", &GraphicsPipelineBuilder::uniform_buffer},
             {"storage_buffer", &GraphicsPipelineBuilder::storage_buffer},
             // count>1 declares a descriptor array: one binding holding N textures,
             // written with set_image(..., index=i) and indexed in the shader.
             {"texture", &GraphicsPipelineBuilder::texture},
             {"storage_image", &GraphicsPipelineBuilder::storage_image}})
    {
        // stage= takes one ShaderStage or a sequence (0.30): a binding read
        // by two stages is one call. The loop reuses the C++ merge, so the
        // sequence and the two-call spelling cannot disagree.
        graphics.def(
            name,
            [declarator, name](
                GraphicsPipelineBuilder& self,
                uint32_t binding,
                const py::object& stage,
                uint32_t set,
                uint32_t count,
                std::optional<bool> update_after_bind) -> GraphicsPipelineBuilder&
            {
                for (const ShaderStage s : stages_of(stage, name))
                {
                    (self.*declarator)(binding, s, set, count, update_after_bind);
                }
                return self;
            },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set") = 0,
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none());
    }

    // No stage arguments anywhere: compute has exactly one stage, so asking for
    // it could only ever be redundant or wrong. build() takes no target —
    // compute has no attachments.
    py::class_<ComputePipelineBuilder, std::shared_ptr<ComputePipelineBuilder>> compute(m, "ComputePipelineBuilder");

    // Same four declarators as the graphics builder, minus the stage.
    using ComputeDeclarator =
        ComputePipelineBuilder& (ComputePipelineBuilder::*)(uint32_t, uint32_t, uint32_t, std::optional<bool>);
    for (auto [name, declarator] : std::initializer_list<std::pair<const char*, ComputeDeclarator>>{
             {"uniform_buffer", &ComputePipelineBuilder::uniform_buffer},
             {"storage_buffer", &ComputePipelineBuilder::storage_buffer},
             // A sampled image in a compute shader: filtering, mips and address
             // modes, which a storage image has none of. The declarator was simply
             // missing until 0.21 -- everything downstream already handled it.
             {"texture", &ComputePipelineBuilder::texture},
             {"storage_image", &ComputePipelineBuilder::storage_image}})
    {
        compute.def(
            name,
            declarator,
            py::arg("binding"),
            py::arg("set") = 0,
            py::arg("count") = 1,
            py::arg("update_after_bind") = py::none());
    }

    compute.def("shader", &ComputePipelineBuilder::shader, py::arg("shader"))
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
                {
                    hr->watch_pipeline(pipeline);
                }
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
               uint32_t index,
               std::optional<std::uint32_t> layer,
               std::optional<std::uint32_t> mip)
            {
                require_same_context(self.owner(), image->owner(), "set_image");
                unwrap(self.set_image(binding, std::move(image), std::move(sampler), index, layer, mip), nullptr);
            },
            py::arg("binding"),
            py::arg("image"),
            py::arg("sampler") = py::none(),
            // Keyword-only: set_image(0, img, 3) read as "index 3" and passed 3 as
            // a sampler. Everywhere else in the API the extras are keyword-only,
            // and the sibling verbs follow so the rule stays one rule (0.23).
            // layer=/mip= join them for the same reason, and because two
            // adjacent ints selecting different axes is the trap target.layer()
            // fixed in 0.23.
            py::kw_only(),
            py::arg("index") = 0,
            py::arg("layer") = py::none(),
            py::arg("mip") = py::none())
        .def(
            "set_storage_image",
            [](DescriptorSet& self,
               uint32_t binding,
               std::shared_ptr<Image> image,
               uint32_t index,
               std::optional<std::uint32_t> layer,
               std::optional<std::uint32_t> mip)
            {
                require_same_context(self.owner(), image->owner(), "set_storage_image");
                unwrap(self.set_storage_image(binding, std::move(image), index, layer, mip), nullptr);
            },
            py::arg("binding"),
            py::arg("image"),
            py::kw_only(),
            py::arg("index") = 0,
            py::arg("layer") = py::none(),
            py::arg("mip") = py::none())
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
            [](DescriptorPool& pool, const std::shared_ptr<Pipeline>& pipeline, uint32_t setIndex) -> py::object
            {
                require_same_context(pool.owner(), pipeline->owner(), "allocate_set");
                return py::cast(unwrap(pool.allocate_descriptor_set(pipeline, setIndex), pool.logger().get()));
            },
            py::arg("pipeline"),
            py::arg("set") = 0)
        .def(
            "allocate_frame_set",
            [](DescriptorPool& pool, const std::shared_ptr<Pipeline>& pipeline, uint32_t setIndex) -> py::object
            {
                require_same_context(pool.owner(), pipeline->owner(), "allocate_frame_set");
                return py::cast(unwrap(pool.allocate_frame_descriptor_set(pipeline, setIndex), pool.logger().get()));
            },
            py::arg("pipeline"),
            py::arg("set") = 0);
}
