#include "Bindings.hpp"

void bind_commands(py::module_& m)
{
    // Every recording method returns the command buffer itself, so the two
    // spellings are the same API:
    //     cmd.begin_rendering(t).bind_pipeline(p).draw(3)
    // and the statement-per-line style both work. The lambdas return the
    // shared_ptr self (not the C++ reference) so pybind hands back the SAME
    // Python object — `cmd.draw(3) is cmd`.
    py::class_<CommandBuffer, std::shared_ptr<CommandBuffer>>(m, "CommandBuffer")
        .def(
            "begin",
            [](std::shared_ptr<CommandBuffer> self)
            {
                self->begin();
                return self;
            })
        // The target is required. begin_rendering() silently meaning "the
        // swapchain" made presentation a special case disguised as the default.
        .def(
            "begin_rendering",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<RenderTarget> target,
               const py::object& clear_color,
               float clear_depth,
               std::uint32_t clear_stencil)
            {
                require_same_context(self->owner(), target->owner(), "begin_rendering");
                require_sliced_when_3d(*target, "begin_rendering");
                auto clears = parse_clear_colors(clear_color);
                require_preservable(*target, !clears.has_value(), "begin_rendering");
                self->begin_rendering(std::move(target), clears, clear_depth, clear_stencil);
                return self;
            },
            py::arg("target"),
            py::arg("clear_color") = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f),
            py::arg("clear_depth") = 1.0f,
            py::arg("clear_stencil") = 0)
        .def(
            "end_rendering",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<RenderTarget> target)
            {
                self->end_rendering(std::move(target));
                return self;
            },
            py::arg("target"))
        // With-statement sugar over the same pair: __enter__ records
        // begin_rendering and hands back the cmd, __exit__ records
        // end_rendering unconditionally — the pair cannot be left open.
        .def(
            "rendering",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<RenderTarget> target,
               const py::object& clear_color,
               float clear_depth,
               std::uint32_t clear_stencil)
            {
                require_same_context(self->owner(), target->owner(), "rendering");
                require_sliced_when_3d(*target, "rendering");
                auto clears = parse_clear_colors(clear_color);
                require_preservable(*target, !clears.has_value(), "rendering");
                return RenderingScope{
                    std::move(self), std::move(target), std::move(clears), clear_depth, clear_stencil};
            },
            py::arg("target"),
            py::arg("clear_color") = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f),
            py::arg("clear_depth") = 1.0f,
            py::arg("clear_stencil") = 0)
        // GPU timer: records the opening timestamp and returns a Timer handle.
        // Stop it with t.stop() or a `with` block; read it back with t.ms.
        .def(
            "timer",
            [](std::shared_ptr<CommandBuffer> self)
            {
                const std::size_t index = self->start_timer();
                const std::uint64_t generation = self->recording_generation();
                return std::make_shared<Timer>(Timer{std::move(self), index, generation, false});
            })
        // A named scope in a capture. `with cmd.label("shadow pass"):` is the
        // form to use; begin_label/end_label are the escape hatch for a
        // recording split across functions, exactly as begin_rendering/
        // end_rendering are. end_label ignores an unbalanced close.
        .def(
            "label",
            [](std::shared_ptr<CommandBuffer> self, std::string name)
            { return LabelScope{std::move(self), std::move(name)}; },
            py::arg("name"))
        .def(
            "begin_label",
            [](std::shared_ptr<CommandBuffer> self, const std::string& name)
            {
                self->begin_label(name);
                return self;
            },
            py::arg("name"))
        .def(
            "end_label",
            [](std::shared_ptr<CommandBuffer> self)
            {
                self->end_label();
                return self;
            })
        // Occlusion query: counts the fragments of the draws inside it that
        // passed the depth and stencil tests. Must sit inside a rendering scope.
        .def(
            "occlusion_query",
            [](std::shared_ptr<CommandBuffer> self)
            {
                auto index = unwrap(self->start_occlusion_query(), nullptr);
                const std::uint64_t generation = self->recording_generation();
                return std::make_shared<OcclusionQuery>(OcclusionQuery{std::move(self), index, generation, false});
            })
        // The no-argument versions are gone: begin_rendering emits a full-target
        // viewport and scissor itself. These remain for split-screen and similar.
        .def(
            "set_viewport",
            [](std::shared_ptr<CommandBuffer> self, float x, float y, float width, float height)
            {
                self->set_viewport(x, y, width, height);
                return self;
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("width"),
            py::arg("height"))
        .def(
            "set_scissor",
            [](std::shared_ptr<CommandBuffer> self,
               std::int32_t x,
               std::int32_t y,
               std::uint32_t width,
               std::uint32_t height)
            {
                self->set_scissor(x, y, width, height);
                return self;
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("width"),
            py::arg("height"))
        .def(
            "bind_pipeline",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Pipeline> pipeline)
            {
                require_same_context(self->owner(), pipeline->owner(), "bind_pipeline");
                self->bind_pipeline(std::move(pipeline));
                return self;
            },
            py::arg("pipeline"))
        .def(
            "bind_vertex_buffer",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Buffer> buffer, std::uint32_t binding)
            {
                require_same_context(self->owner(), buffer->owner(), "bind_vertex_buffer");
                self->bind_vertex_buffer(std::move(buffer), binding);
                return self;
            },
            py::arg("buffer"),
            py::arg("binding") = 0)
        .def(
            "bind_index_buffer",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Buffer> buffer)
            {
                require_same_context(self->owner(), buffer->owner(), "bind_index_buffer");
                self->bind_index_buffer(std::move(buffer));
                return self;
            },
            py::arg("buffer"))
        .def(
            "draw",
            [](std::shared_ptr<CommandBuffer> self, uint32_t vertex_count, uint32_t instances)
            {
                self->draw(vertex_count, instances);
                return self;
            },
            py::arg("vertex_count"),
            py::arg("instances") = 1)
        .def(
            "draw_indexed",
            [](std::shared_ptr<CommandBuffer> self,
               uint32_t index_count,
               uint32_t first_index,
               int32_t vertex_offset,
               uint32_t instances)
            {
                self->draw_indexed(index_count, first_index, vertex_offset, instances);
                return self;
            },
            py::arg("index_count"),
            py::arg("first_index") = 0,
            py::arg("vertex_offset") = 0,
            py::arg("instances") = 1)
        .def(
            "dispatch",
            [](std::shared_ptr<CommandBuffer> self,
               uint32_t group_count_x,
               uint32_t group_count_y,
               uint32_t group_count_z)
            {
                self->dispatch(group_count_x, group_count_y, group_count_z);
                return self;
            },
            py::arg("group_count_x"),
            py::arg("group_count_y") = 1,
            py::arg("group_count_z") = 1)
        // Indirect draw/dispatch: the arguments come out of a storage buffer the
        // GPU can write, so a compute pass decides what gets drawn. Chaining is
        // preserved (return self) even though these are fallible — unwrap raises,
        // and a successful call keeps reading like every other recording verb.
        .def(
            "draw_indirect",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Buffer> buffer,
               VkDeviceSize offset,
               std::uint32_t count,
               std::shared_ptr<Buffer> count_buffer,
               VkDeviceSize count_offset)
            {
                require_same_context(self->owner(), buffer->owner(), "draw_indirect");
                if (count_buffer)
                {
                    require_same_context(self->owner(), count_buffer->owner(), "draw_indirect");
                }
                unwrap(
                    self->draw_indirect(std::move(buffer), offset, count, std::move(count_buffer), count_offset),
                    nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("offset") = 0,
            py::arg("count") = 1,
            py::arg("count_buffer") = py::none(),
            py::arg("count_offset") = 0)
        .def(
            "draw_indexed_indirect",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Buffer> buffer,
               VkDeviceSize offset,
               std::uint32_t count,
               std::shared_ptr<Buffer> count_buffer,
               VkDeviceSize count_offset)
            {
                require_same_context(self->owner(), buffer->owner(), "draw_indexed_indirect");
                if (count_buffer)
                {
                    require_same_context(self->owner(), count_buffer->owner(), "draw_indexed_indirect");
                }
                unwrap(
                    self->draw_indexed_indirect(
                        std::move(buffer), offset, count, std::move(count_buffer), count_offset),
                    nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("offset") = 0,
            py::arg("count") = 1,
            py::arg("count_buffer") = py::none(),
            py::arg("count_offset") = 0)
        .def(
            "dispatch_indirect",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Buffer> buffer, VkDeviceSize offset)
            {
                require_same_context(self->owner(), buffer->owner(), "dispatch_indirect");
                unwrap(self->dispatch_indirect(std::move(buffer), offset), nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("offset") = 0)
        .def(
            "barrier",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Buffer> buffer, Access src, Access dst)
            {
                require_same_context(self->owner(), buffer->owner(), "barrier");
                unwrap(self->barrier(std::move(buffer), src, dst), nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("src"),
            py::arg("dst"))
        .def(
            "barrier",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Image> image, Access src, Access dst)
            {
                require_same_context(self->owner(), image->owner(), "barrier");
                unwrap(self->barrier(std::move(image), src, dst), nullptr);
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
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Image> image, Access src)
            {
                require_same_context(self->owner(), image->owner(), "generate_mipmaps");
                unwrap(self->generate_mipmaps(std::move(image), src), nullptr);
                return self;
            },
            py::arg("image"),
            py::kw_only(),
            py::arg("src") = Access::SHADER_READ)
        .def(
            "copy_image",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Image> src,
               std::shared_ptr<Image> dst,
               Access src_access)
            {
                require_same_context(self->owner(), src->owner(), "copy_image");
                require_same_context(self->owner(), dst->owner(), "copy_image");
                unwrap(self->copy_image(std::move(src), std::move(dst), src_access), nullptr);
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
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Image> src,
               std::shared_ptr<Image> dst,
               Access src_access,
               Filter filter)
            {
                require_same_context(self->owner(), src->owner(), "blit_image");
                require_same_context(self->owner(), dst->owner(), "blit_image");
                unwrap(self->blit_image(std::move(src), std::move(dst), src_access, to_vk_filter(filter)), nullptr);
                return self;
            },
            py::arg("src"),
            py::arg("dst"),
            py::kw_only(),
            py::arg("src_access") = Access::SHADER_READ,
            py::arg("filter") = Filter::LINEAR)
        .def(
            "copy_buffer",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Buffer> src,
               std::shared_ptr<Buffer> dst,
               VkDeviceSize src_offset,
               VkDeviceSize dst_offset,
               VkDeviceSize size)
            {
                require_same_context(self->owner(), src->owner(), "copy_buffer");
                require_same_context(self->owner(), dst->owner(), "copy_buffer");
                unwrap(self->copy_buffer(std::move(src), std::move(dst), src_offset, dst_offset, size), nullptr);
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
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Buffer> buffer,
               std::uint32_t value,
               VkDeviceSize offset,
               VkDeviceSize size)
            {
                require_same_context(self->owner(), buffer->owner(), "fill_buffer");
                unwrap(self->fill_buffer(std::move(buffer), value, offset, size), nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("value") = 0,
            py::kw_only(),
            py::arg("offset") = 0,
            py::arg("size") = 0)
        .def(
            "clear_image",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Image> image, const py::object& color)
            {
                require_same_context(self->owner(), image->owner(), "clear_image");
                std::array<float, 4> rgba{0.0f, 0.0f, 0.0f, 1.0f};
                py::sequence seq = py::cast<py::sequence>(color);
                for (std::size_t i = 0; i < 4 && i < py::len(seq); ++i)
                {
                    rgba[i] = py::cast<float>(seq[i]);
                }
                unwrap(self->clear_image(std::move(image), rgba), nullptr);
                return self;
            },
            py::arg("image"),
            py::arg("color") = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f))
        // No stage argument: the Pipeline already records which stages its push
        // constant range covers, so repeating it could only ever be wrong.
        .def(
            "push_constants",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Pipeline> pipeline,
               uint32_t offset,
               std::string_view data)
            {
                require_same_context(self->owner(), pipeline->owner(), "push_constants");
                self->push_constants(std::move(pipeline), offset, static_cast<uint32_t>(data.size()), data.data());
                return self;
            },
            py::arg("pipeline"),
            py::arg("offset"),
            py::arg("data"))
        .def(
            "bind_descriptor_set",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<DescriptorSet> descriptor_set,
               std::shared_ptr<Pipeline> pipeline,
               uint32_t set)
            {
                require_same_context(self->owner(), descriptor_set->owner(), "bind_descriptor_set");
                require_same_context(self->owner(), pipeline->owner(), "bind_descriptor_set");
                self->bind_descriptor_set(std::move(descriptor_set), std::move(pipeline), set);
                return self;
            },
            py::arg("descriptor_set"),
            py::arg("pipeline"),
            py::arg("set"));

    py::class_<RenderingScope>(m, "RenderingScope")
        .def(
            "__enter__",
            [](RenderingScope& self)
            {
                self.cmd->begin_rendering(self.target, self.clear_color, self.clear_depth, self.clear_stencil);
                return self.cmd;
            })
        .def(
            "__exit__",
            [](RenderingScope& self, py::object, py::object, py::object)
            {
                self.cmd->end_rendering(self.target);
                return false; // never swallow exceptions
            });

    py::class_<LabelScope>(m, "LabelScope")
        .def(
            "__enter__",
            [](LabelScope& self)
            {
                self.cmd->begin_label(self.name);
                return self.cmd;
            })
        .def(
            "__exit__",
            [](LabelScope& self, py::object, py::object, py::object)
            {
                self.cmd->end_label();
                return false; // never swallow exceptions
            });

    py::class_<OcclusionQuery, std::shared_ptr<OcclusionQuery>>(m, "OcclusionQuery")
        .def("stop", [](OcclusionQuery& self) { self.stop(); })
        .def("__enter__", [](std::shared_ptr<OcclusionQuery> self) { return self; })
        .def(
            "__exit__",
            [](OcclusionQuery& self, py::object, py::object, py::object)
            {
                self.stop();
                return false; // never swallow exceptions
            })
        // None means one thing now: the submit has not finished. A stale handle
        // raises instead — see the Timer below for the argument, which is the
        // same one.
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

    py::class_<Timer, std::shared_ptr<Timer>>(m, "Timer")
        .def("stop", [](Timer& self) { self.stop(); })
        .def("__enter__", [](std::shared_ptr<Timer> self) { return self; })
        .def(
            "__exit__",
            [](Timer& self, py::object, py::object, py::object)
            {
                self.stop();
                return false; // never swallow exceptions
            })
        // Three answers, three shapes. UnsupportedError when the device has no
        // usable timestamps, StateError when the handle predates a begin(), and
        // None only for "the submit is still running". They used to be one
        // nullopt, and a caller could not tell "wait longer" from "this GPU
        // cannot" — which are opposite reactions.
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
