#include "Bindings.hpp"

namespace
{
    // The half of a QueryHandle binding the two instantiations share; the caller
    // chains the reading property, which is the only difference between them.
    template <typename Handle>
    py::class_<Handle, std::shared_ptr<Handle>> bind_query_handle(py::module_& m, const char* name)
    {
        return py::class_<Handle, std::shared_ptr<Handle>>(m, name)
            .def("stop", [](Handle& self) { self.stop(); })
            .def("__enter__", [](std::shared_ptr<Handle> self) { return self; })
            .def(
                "__exit__",
                [](Handle& self, const py::object&, const py::object&, const py::object&)
                {
                    self.stop();
                    return false; // never swallow exceptions
                });
    }

    // Every recording verb passes the same gate first: the pass is live, not
    // sealed, and the right kind for the verb. Raises so the verb bodies read
    // straight.
    void guard(const Pass& pass, Pass::VerbScope scope, const char* verb)
    {
        unwrap(pass.guard(scope, verb), nullptr);
    }
} // namespace

void bind_commands(py::module_& m)
{
    // Every recording method returns the pass itself, so the two spellings are
    // the same API:
    //     p.bind_pipeline(pipe).bind_vertex_buffer(vbuf).draw(3)
    // and the statement-per-line style both work. The lambdas return the
    // shared_ptr self (not the C++ reference) so pybind hands back the SAME
    // Python object — `p.draw(3) is p`.
    auto pass = py::class_<Pass, std::shared_ptr<Pass>>(m, "Pass");
    pass
        // The pass is a handle; `with` is sugar that seals it early. __exit__
        // deliberately does NOT submit — who submits the graph is still the
        // caller's decision.
        .def("__enter__", [](std::shared_ptr<Pass> self) { return self; })
        .def(
            "__exit__",
            [](Pass& self, const py::object&, const py::object&, const py::object&)
            {
                self.seal();
                return false; // never swallow exceptions
            })
        // Toggling a pass recompiles the graph without re-recording anything:
        // a disabled pass is absent from the compile, and whatever it wrote is
        // stale for the passes after it — the caller's responsibility, exactly
        // as a submit it chose to skip.
        .def_property("enabled", &Pass::enabled, &Pass::set_enabled)
        .def_property_readonly("name", &Pass::name)
        .def(
            "bind_pipeline",
            [](std::shared_ptr<Pass> self, const std::shared_ptr<Pipeline>& pipeline)
            {
                guard(*self, Pass::VerbScope::Any, "bind_pipeline");
                require_same_context(self->recorder().owner(), pipeline->owner(), "bind_pipeline");
                self->recorder().bind_pipeline(pipeline);
                return self;
            },
            py::arg("pipeline"))
        .def(
            "bind_vertex_buffer",
            [](std::shared_ptr<Pass> self, const std::shared_ptr<Buffer>& buffer, std::uint32_t binding)
            {
                guard(*self, Pass::VerbScope::Render, "bind_vertex_buffer");
                require_same_context(self->recorder().owner(), buffer->owner(), "bind_vertex_buffer");
                self->recorder().bind_vertex_buffer(buffer, binding);
                return self;
            },
            py::arg("buffer"),
            py::arg("binding") = 0)
        .def(
            "bind_index_buffer",
            [](std::shared_ptr<Pass> self, const std::shared_ptr<Buffer>& buffer)
            {
                guard(*self, Pass::VerbScope::Render, "bind_index_buffer");
                require_same_context(self->recorder().owner(), buffer->owner(), "bind_index_buffer");
                self->recorder().bind_index_buffer(buffer);
                return self;
            },
            py::arg("buffer"))
        .def(
            "draw",
            [](std::shared_ptr<Pass> self, uint32_t vertex_count, uint32_t instances)
            {
                guard(*self, Pass::VerbScope::Render, "draw");
                self->recorder().draw(vertex_count, instances);
                return self;
            },
            py::arg("vertex_count"),
            py::arg("instances") = 1)
        .def(
            "draw_indexed",
            [](std::shared_ptr<Pass> self,
               uint32_t index_count,
               uint32_t first_index,
               int32_t vertex_offset,
               uint32_t instances)
            {
                guard(*self, Pass::VerbScope::Render, "draw_indexed");
                self->recorder().draw_indexed(index_count, first_index, vertex_offset, instances);
                return self;
            },
            py::arg("index_count"),
            py::arg("first_index") = 0,
            py::arg("vertex_offset") = 0,
            py::arg("instances") = 1)
        // A dispatch inside a rendering scope is illegal in Vulkan, and a
        // render pass IS one rendering scope — so the pass kind decides at
        // record time, with a message naming the fix, where the old API left
        // it to a replay-time validation error.
        .def(
            "dispatch",
            [](std::shared_ptr<Pass> self, uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z)
            {
                guard(*self, Pass::VerbScope::General, "dispatch");
                self->recorder().dispatch(group_count_x, group_count_y, group_count_z);
                return self;
            },
            py::arg("group_count_x"),
            py::arg("group_count_y") = 1,
            py::arg("group_count_z") = 1)
        // The no-argument versions are gone since 0.16: a render pass emits a
        // full-target viewport and scissor itself. These remain for
        // split-screen and similar.
        .def(
            "set_viewport",
            [](std::shared_ptr<Pass> self, float x, float y, float width, float height)
            {
                guard(*self, Pass::VerbScope::Render, "set_viewport");
                self->recorder().set_viewport(x, y, width, height);
                return self;
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("width"),
            py::arg("height"))
        .def(
            "set_scissor",
            [](std::shared_ptr<Pass> self, std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height)
            {
                guard(*self, Pass::VerbScope::Render, "set_scissor");
                self->recorder().set_scissor(x, y, width, height);
                return self;
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("width"),
            py::arg("height"))
        // GPU timer: records the opening timestamp and returns a Timer handle.
        // Stop it with t.stop() or a `with` block; read it back with t.ms.
        .def(
            "timer",
            [](std::shared_ptr<Pass> self)
            {
                guard(*self, Pass::VerbScope::Any, "timer");
                CommandBuffer& rec = self->recorder();
                const std::size_t index = rec.start_timer();
                const std::uint64_t generation = rec.recording_generation();
                auto cmd = std::shared_ptr<CommandBuffer>(self, &rec);
                return std::make_shared<Timer>(Timer{std::move(cmd), index, generation, false});
            })
        // A named scope in a capture. `with p.label("shadow pass"):` is the
        // form to use; begin_label/end_label are the escape hatch for a
        // recording split across functions. end_label ignores an unbalanced
        // close.
        .def(
            "label",
            [](std::shared_ptr<Pass> self, std::string name)
            {
                guard(*self, Pass::VerbScope::Any, "label");
                return LabelScope{.pass = std::move(self), .name = std::move(name)};
            },
            py::arg("name"))
        .def(
            "begin_label",
            [](std::shared_ptr<Pass> self, const std::string& name)
            {
                guard(*self, Pass::VerbScope::Any, "begin_label");
                self->recorder().begin_label(name);
                return self;
            },
            py::arg("name"))
        .def(
            "end_label",
            [](std::shared_ptr<Pass> self)
            {
                guard(*self, Pass::VerbScope::Any, "end_label");
                self->recorder().end_label();
                return self;
            })
        // Occlusion query: counts the fragments of the draws inside it that
        // passed the depth and stencil tests. Vulkan requires it to begin and
        // end within one render pass, which the pass kind now says up front.
        .def(
            "occlusion_query",
            [](std::shared_ptr<Pass> self)
            {
                guard(*self, Pass::VerbScope::Render, "occlusion_query");
                CommandBuffer& rec = self->recorder();
                auto index = unwrap(rec.start_occlusion_query(), nullptr);
                const std::uint64_t generation = rec.recording_generation();
                auto cmd = std::shared_ptr<CommandBuffer>(self, &rec);
                return std::make_shared<OcclusionQuery>(OcclusionQuery{std::move(cmd), index, generation, false});
            });
    // Indirect draw/dispatch: the arguments come out of a storage buffer the
    // GPU can write, so a compute pass decides what gets drawn. Chaining is
    // preserved (return self) even though these are fallible — unwrap raises,
    // and a successful call keeps reading like every other recording verb.
    //
    // The two draw verbs are one loop: same signature, same guards, only the
    // member called differs.
    using IndirectDraw = std::expected<void, Error> (CommandBuffer::*)(
        std::shared_ptr<Buffer>, VkDeviceSize, std::uint32_t, std::shared_ptr<Buffer>, VkDeviceSize, std::uint32_t);
    for (auto [name, verb] : std::initializer_list<std::pair<const char*, IndirectDraw>>{
             {"draw_indirect", &CommandBuffer::draw_indirect},
             {"draw_indexed_indirect", &CommandBuffer::draw_indexed_indirect}})
    {
        pass.def(
            name,
            [name, verb](
                std::shared_ptr<Pass> self,
                std::shared_ptr<Buffer> buffer,
                VkDeviceSize offset,
                std::uint32_t count,
                std::shared_ptr<Buffer> count_buffer,
                VkDeviceSize count_offset,
                std::uint32_t stride)
            {
                guard(*self, Pass::VerbScope::Render, name);
                require_same_context(self->recorder().owner(), buffer->owner(), name);
                if (count_buffer)
                {
                    require_same_context(self->recorder().owner(), count_buffer->owner(), name);
                }
                unwrap(
                    (self->recorder().*
                     verb)(std::move(buffer), offset, count, std::move(count_buffer), count_offset, stride),
                    nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("offset") = 0,
            py::arg("count") = 1,
            py::arg("count_buffer") = py::none(),
            py::arg("count_offset") = 0,
            py::arg("stride") = 0);
    }
    pass.def(
            "dispatch_indirect",
            [](std::shared_ptr<Pass> self, std::shared_ptr<Buffer> buffer, VkDeviceSize offset)
            {
                guard(*self, Pass::VerbScope::General, "dispatch_indirect");
                require_same_context(self->recorder().owner(), buffer->owner(), "dispatch_indirect");
                unwrap(self->recorder().dispatch_indirect(std::move(buffer), offset), nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("offset") = 0)
        // Manual barriers: the escape hatch inside the graph. A pass created
        // with auto_barriers=False computes nothing for itself, and these are
        // how its hazards are expressed — the graph's compile still sees them,
        // so the automatic passes around it order correctly.
        .def(
            "barrier",
            [](std::shared_ptr<Pass> self, std::shared_ptr<Buffer> buffer, Access src, Access dst)
            {
                guard(*self, Pass::VerbScope::General, "barrier");
                require_same_context(self->recorder().owner(), buffer->owner(), "barrier");
                unwrap(self->recorder().barrier(std::move(buffer), src, dst), nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("src"),
            py::arg("dst"))
        .def(
            "barrier",
            [](std::shared_ptr<Pass> self, std::shared_ptr<Image> image, Access src, Access dst)
            {
                guard(*self, Pass::VerbScope::General, "barrier");
                require_same_context(self->recorder().owner(), image->owner(), "barrier");
                unwrap(self->recorder().barrier(std::move(image), src, dst), nullptr);
                return self;
            },
            py::arg("image"),
            py::arg("src"),
            py::arg("dst"))
        // Fill mip levels 1..N of a mipped image from mip 0 (all layers). `src`
        // names mip 0's current layout: SHADER_READ (default, an uploaded/baked
        // image) or SHADER_WRITE (mip 0 fresh from compute imageStore).
        .def(
            "generate_mipmaps",
            [](std::shared_ptr<Pass> self, std::shared_ptr<Image> image, Access src)
            {
                guard(*self, Pass::VerbScope::General, "generate_mipmaps");
                require_same_context(self->recorder().owner(), image->owner(), "generate_mipmaps");
                unwrap(self->recorder().generate_mipmaps(std::move(image), src), nullptr);
                return self;
            },
            py::arg("image"),
            py::kw_only(),
            py::arg("src") = Access::SHADER_READ)
        .def(
            "copy_image",
            [](std::shared_ptr<Pass> self, std::shared_ptr<Image> src, std::shared_ptr<Image> dst, Access src_access)
            {
                guard(*self, Pass::VerbScope::General, "copy_image");
                require_same_context(self->recorder().owner(), src->owner(), "copy_image");
                require_same_context(self->recorder().owner(), dst->owner(), "copy_image");
                unwrap(self->recorder().copy_image(std::move(src), std::move(dst), src_access), nullptr);
                return self;
            },
            py::arg("src"),
            py::arg("dst"),
            py::kw_only(),
            py::arg("src_access") = Access::SHADER_READ)
        // The resizing sibling of copy_image. `filter` reuses bz.Filter, which
        // the sampler already introduced — the question "how do you sample when
        // the sizes differ" has one answer in this library, not two enums.
        .def(
            "blit_image",
            [](std::shared_ptr<Pass> self,
               std::shared_ptr<Image> src,
               std::shared_ptr<Image> dst,
               Access src_access,
               Filter filter)
            {
                guard(*self, Pass::VerbScope::General, "blit_image");
                require_same_context(self->recorder().owner(), src->owner(), "blit_image");
                require_same_context(self->recorder().owner(), dst->owner(), "blit_image");
                unwrap(
                    self->recorder().blit_image(std::move(src), std::move(dst), src_access, to_vk_filter(filter)),
                    nullptr);
                return self;
            },
            py::arg("src"),
            py::arg("dst"),
            py::kw_only(),
            py::arg("src_access") = Access::SHADER_READ,
            py::arg("filter") = Filter::LINEAR)
        .def(
            "copy_buffer",
            [](std::shared_ptr<Pass> self,
               std::shared_ptr<Buffer> src,
               std::shared_ptr<Buffer> dst,
               VkDeviceSize src_offset,
               VkDeviceSize dst_offset,
               VkDeviceSize size)
            {
                guard(*self, Pass::VerbScope::General, "copy_buffer");
                require_same_context(self->recorder().owner(), src->owner(), "copy_buffer");
                require_same_context(self->recorder().owner(), dst->owner(), "copy_buffer");
                unwrap(
                    self->recorder().copy_buffer(std::move(src), std::move(dst), src_offset, dst_offset, size),
                    nullptr);
                return self;
            },
            py::arg("src"),
            py::arg("dst"),
            py::kw_only(),
            py::arg("src_offset") = 0,
            py::arg("dst_offset") = 0,
            py::arg("size") = 0)
        .def(
            "fill_buffer",
            [](std::shared_ptr<Pass> self,
               std::shared_ptr<Buffer> buffer,
               std::uint32_t value,
               VkDeviceSize offset,
               VkDeviceSize size)
            {
                guard(*self, Pass::VerbScope::General, "fill_buffer");
                require_same_context(self->recorder().owner(), buffer->owner(), "fill_buffer");
                unwrap(self->recorder().fill_buffer(std::move(buffer), value, offset, size), nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("value") = 0,
            py::kw_only(),
            py::arg("offset") = 0,
            py::arg("size") = 0)
        .def(
            "clear_image",
            [](std::shared_ptr<Pass> self, std::shared_ptr<Image> image, const py::object& color)
            {
                guard(*self, Pass::VerbScope::General, "clear_image");
                require_same_context(self->recorder().owner(), image->owner(), "clear_image");
                std::array<float, 4> rgba{0.0f, 0.0f, 0.0f, 1.0f};
                auto seq = py::cast<py::sequence>(color);
                for (std::size_t i = 0; i < 4 && i < py::len(seq); ++i)
                {
                    rgba[i] = py::cast<float>(seq[i]);
                }
                unwrap(self->recorder().clear_image(std::move(image), rgba), nullptr);
                return self;
            },
            py::arg("image"),
            py::arg("color") = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f))
        // No stage argument: the Pipeline already records which stages its push
        // constant range covers, so repeating it could only ever be wrong.
        .def(
            "push_constants",
            [](std::shared_ptr<Pass> self,
               const std::shared_ptr<Pipeline>& pipeline,
               uint32_t offset,
               std::string_view data)
            {
                guard(*self, Pass::VerbScope::Any, "push_constants");
                require_same_context(self->recorder().owner(), pipeline->owner(), "push_constants");
                self->recorder().push_constants(pipeline, offset, static_cast<uint32_t>(data.size()), data.data());
                return self;
            },
            py::arg("pipeline"),
            py::arg("offset"),
            py::arg("data"))
        // The short form: the bound pipeline is the pipeline (0.25). Distinguished
        // from the long one by the type of the first argument, which is how
        // create_image and create_buffer already pick an overload.
        .def(
            "push_constants",
            [](std::shared_ptr<Pass> self, uint32_t offset, std::string_view data)
            {
                guard(*self, Pass::VerbScope::Any, "push_constants");
                unwrap(
                    self->recorder().push_constants(offset, static_cast<uint32_t>(data.size()), data.data()), nullptr);
                return self;
            },
            py::arg("offset"),
            py::arg("data"))
        .def(
            "bind_descriptor_set",
            [](std::shared_ptr<Pass> self,
               const std::shared_ptr<DescriptorSet>& descriptor_set,
               const std::shared_ptr<Pipeline>& pipeline,
               uint32_t set)
            {
                guard(*self, Pass::VerbScope::Any, "bind_descriptor_set");
                require_same_context(self->recorder().owner(), descriptor_set->owner(), "bind_descriptor_set");
                require_same_context(self->recorder().owner(), pipeline->owner(), "bind_descriptor_set");
                self->recorder().bind_descriptor_set(descriptor_set, pipeline, set);
                return self;
            },
            py::arg("descriptor_set"),
            py::arg("pipeline"),
            py::arg("set") = 0)
        // The short form (0.25): the set knows which index it was allocated for
        // and at which bind point, and bind_pipeline already recorded the
        // pipeline. Registered after the long one, which pybind tries first — a
        // call with three arguments cannot match this signature, so the two
        // cannot collide.
        .def(
            "bind_descriptor_set",
            [](std::shared_ptr<Pass> self, const std::shared_ptr<DescriptorSet>& descriptor_set)
            {
                guard(*self, Pass::VerbScope::Any, "bind_descriptor_set");
                require_same_context(self->recorder().owner(), descriptor_set->owner(), "bind_descriptor_set");
                unwrap(self->recorder().bind_descriptor_set(descriptor_set), nullptr);
                return self;
            },
            py::arg("descriptor_set"));

    py::class_<LabelScope>(m, "LabelScope")
        .def(
            "__enter__",
            [](LabelScope& self)
            {
                self.pass->recorder().begin_label(self.name);
                return self.pass;
            })
        .def(
            "__exit__",
            [](LabelScope& self, const py::object&, const py::object&, const py::object&)
            {
                self.pass->recorder().end_label();
                return false; // never swallow exceptions
            });

    // The stop/__enter__/__exit__ trio is the shared QueryHandle contract; the
    // reading property below each call is the only difference between the two.
    //
    // Three answers, three shapes, on both readers. UnsupportedError when the
    // device cannot answer at all, StateError when the handle predates a
    // reset, and None only for "the submit is still running". They used to be
    // one nullopt, and a caller could not tell "wait longer" from "this GPU
    // cannot" — which are opposite reactions.
    bind_query_handle<OcclusionQuery>(m, "OcclusionQuery")
        .def_property_readonly(
            "samples",
            [](const OcclusionQuery& self) -> py::object
            {
                const auto reading = self.cmd->read_occlusion_query(self.index, self.generation);
                raise_for_query_status(reading.status, "OcclusionQuery.samples");
                if (reading.status != QueryStatus::Ok)
                {
                    return py::none();
                }
                return py::cast(reading.samples);
            });

    bind_query_handle<Timer>(m, "Timer")
        .def_property_readonly(
            "ms",
            [](const Timer& self) -> py::object
            {
                const auto reading = self.cmd->read_timer(self.index, self.generation);
                raise_for_query_status(reading.status, "Timer.ms");
                if (reading.status != QueryStatus::Ok)
                {
                    return py::none();
                }
                return py::cast(reading.ms);
            });
}
