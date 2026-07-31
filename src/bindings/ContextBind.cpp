#include "Bindings.hpp"

void bind_context(py::module_& m)
{
    // Read-only data, so plain attributes: nothing here is a handle and there is
    // nothing to keep alive. Bytes rather than megabytes, because a rounded
    // number cannot be un-rounded and "is it growing" is the question this
    // answers.
    py::class_<Context::MemoryStats>(m, "MemoryStats")
        .def_readonly("used", &Context::MemoryStats::used)
        .def_readonly("reserved", &Context::MemoryStats::reserved)
        .def_readonly("budget", &Context::MemoryStats::budget)
        .def(
            "__repr__",
            [](const Context::MemoryStats& self)
            {
                constexpr double mb = 1024.0 * 1024.0;
                return std::format(
                    "MemoryStats(used={:.1f} MB, reserved={:.1f} MB, budget={:.1f} MB)",
                    static_cast<double>(self.used) / mb,
                    static_cast<double>(self.reserved) / mb,
                    static_cast<double>(self.budget) / mb);
            });

    py::class_<Context, std::shared_ptr<Context>>(m, "Context")
        .def(
            py::init(
                [](std::shared_ptr<Logger> logger,
                   const std::string& validation,
                   std::vector<Feature> features,
                   std::vector<Feature> optional,
                   std::uint32_t frames_in_flight,
                   std::optional<Device> device,
                   std::vector<std::string> raw_extensions,
                   bool auto_barriers,
                   bool hot_reload,
                   bool gpu_timing,
                   bool shader_printf)
                {
                    // ValueError: 1..4 is a fixed range in the signature, so the
                    // argument is wrong on its own — no device, no resource. It must
                    // also stay out of the BazaltError hierarchy, or
                    // `except bz.InitializationError` (fall back to headless) would
                    // swallow a typo in the kwargs. See DESIGN.md.
                    if (frames_in_flight < 1 || frames_in_flight > 4)
                    {
                        throw py::value_error(
                            std::format("frames_in_flight must be between 1 and 4, got {}", frames_in_flight));
                    }

                    ContextConfig config;
                    config.validation = parse_validation(validation);
                    config.required = std::move(features);
                    config.optional = std::move(optional);
                    config.frames_in_flight = frames_in_flight;
                    if (device)
                    {
                        config.device = device->uuid;
                    }
                    config.raw_extensions = std::move(raw_extensions);
                    config.auto_barriers = auto_barriers;
                    config.gpu_timing = gpu_timing;
                    config.shader_printf = shader_printf;

                    if (!logger)
                    {
                        logger = make_default_logger();
                    }
                    auto res = Context::create(logger, config);
                    if (!res)
                    {
                        logger->log(res.error());
                        raise_error(res.error());
                    }
                    auto context = std::move(res.value());
                    // Eagerly, not on the first load_image: every upload counts
                    // towards upload_progress and ctx.wait(), including the ones
                    // that never reach the worker, so there must be exactly one
                    // place that counts them and it must always exist. The worker
                    // thread parks on a condition variable until something arrives.
                    context->set_upload_manager(std::make_unique<UploadManager>(*context));
                    // One kwarg covers both shaders and images: it's one feature, "watch
                    // what you loaded". The watcher holds only weak refs, so it never
                    // keeps a resource alive.
                    if (hot_reload)
                    {
                        context->set_hot_reload(std::make_unique<HotReloadWatcher>(*context));
                    }
                    return context;
                }),
            py::arg("logger") = py::none(),
            py::arg("validation") = "auto",
            py::arg("features") = std::vector<Feature>{},
            py::arg("optional") = std::vector<Feature>{},
            py::arg("frames_in_flight") = 2,
            py::arg("device") = py::none(),
            py::arg("raw_extensions") = std::vector<std::string>{},
            py::arg("auto_barriers") = true,
            py::arg("hot_reload") = false,
            py::arg("gpu_timing") = false,
            py::arg("shader_printf") = false)
        .def_property_readonly("auto_barriers", &Context::auto_barriers)
        .def_property_readonly("shader_printf", &Context::shader_printf)
        .def("memory_stats", &Context::memory_stats)
        .def_property_readonly("subgroup_size", &Context::subgroup_size)
        .def_property_readonly("frames_in_flight", &Context::frames_in_flight)
        // The frame verb of a windowed loop: opens one logical frame for every
        // window on this Context. Advances the ring slot that CommandBuffer,
        // DynamicBuffer and the per-frame descriptor sets index, applies pending
        // hot reloads and reclaims deferred handles — all Context-owned, hence
        // once per frame rather than once per window.
        .def("begin_frame", &Context::begin_frame)
        .def_property_readonly("frame_index", &Context::frame_index)
        .def_property_readonly("logger", &Context::logger)
        .def("supports", &Context::supports, py::arg("feature"))
        .def("max_samples", &Context::max_samples)
        .def_property_readonly("device_name", &Context::device_name)
        .def_property_readonly(
            "api_version", [](const Context& self) { return api_version_string(self.api_version()); })
        .def_property_readonly("headless", &Context::headless)
        .def(
            "create_buffer",
            [](Context& self,
               py::list list,
               BufferType type,
               MemoryUsage usage,
               std::optional<DataType> dataType,
               const std::string& name) -> py::object
            {
                if (list.empty())
                {
                    raise_error(err_resource("Cannot create buffer from empty list"));
                }

                DataType actualType =
                    resolve_data_type(list, dataType, type == BufferType::INDEX ? DataType::UINT32 : DataType::INT32);

                auto buffer = with_list_bytes(
                    list,
                    actualType,
                    [&](const void* data, size_t nbytes)
                    { return unwrap(Buffer::create(self, data, nbytes, type, usage), self.logger().get()); });
                // Recorded so bind_index_buffer can pick VK_INDEX_TYPE_UINT16 vs UINT32
                // instead of assuming.
                buffer->set_data_type(actualType);
                name_buffer(self, buffer, name);
                return py::cast(buffer);
            },
            // One name across the three overloads, so the keyword spelling works
            // whichever body the argument picks — and `list` shadowed a builtin (0.23).
            py::arg("data"),
            py::arg("type"),
            py::arg("usage"),
            py::arg("data_type") = py::none(),
            py::kw_only(),
            py::arg("name") = "")
        .def(
            "create_buffer",
            [](Context& self, py::buffer b, BufferType type, MemoryUsage usage, const std::string& name) -> py::object
            {
                py::buffer_info info = b.request();
                auto buffer = unwrap(
                    Buffer::create(self, info.ptr, contiguous_nbytes(info, "create_buffer"), type, usage),
                    self.logger().get());
                name_buffer(self, buffer, name);
                return py::cast(buffer);
            },
            py::arg("data"),
            py::arg("type"),
            py::arg("usage"),
            py::kw_only(),
            py::arg("name") = "")
        .def(
            "create_buffer",
            [](Context& self, size_t size_in_bytes, BufferType type, MemoryUsage usage, const std::string& name)
                -> py::object
            {
                auto buffer = unwrap(Buffer::create(self, nullptr, size_in_bytes, type, usage), self.logger().get());
                name_buffer(self, buffer, name);
                return py::cast(buffer);
            },
            py::arg("data"),
            py::arg("type"),
            py::arg("usage"),
            py::kw_only(),
            py::arg("name") = "")
        .def(
            "graphics_pipeline",
            [](Context& self) -> std::shared_ptr<GraphicsPipelineBuilder>
            { return std::make_shared<GraphicsPipelineBuilder>(self); })
        .def(
            "compute_pipeline",
            [](Context& self) -> std::shared_ptr<ComputePipelineBuilder>
            { return std::make_shared<ComputePipelineBuilder>(self); })
        .def(
            "compile_shader",
            [](Context& self,
               const std::string& path,
               ShaderStage stage,
               const py::object& source,
               const std::vector<std::string>& include_dirs,
               const std::string& entry_point) -> py::object
            {
                // Only file-backed shaders are watchable: a source= virtual name may
                // not exist on disk, and .spv is recompiled from its own path too.
                const bool from_file = source.is_none();
                auto module = unwrap(
                    ShaderCompiler::compile(self, path, stage, parse_shader_source(source), include_dirs, entry_point),
                    self.logger().get());
                if (auto* hr = self.hot_reload(); hr && from_file && std::filesystem::exists(path))
                {
                    hr->watch_shader(module);
                }
                return py::cast(module);
            },
            py::arg("path"),
            py::arg("stage"),
            py::kw_only(),
            py::arg("source") = py::none(),
            py::arg("include_dirs") = py::tuple(),
            py::arg("entry_point") = "")
        // Encoded bytes instead of a path: a PNG off the network, out of a zip,
        // or straight from PIL. Everything after the decode is the file path's
        // path, hot reload excepted — bazalt has nothing to watch.
        //
        // MUST be declared before the `str` overload would see it: pybind
        // converts both str and bytes to std::string, so a py::bytes argument
        // would otherwise arrive at the path overload and be reported as a
        // missing file. Same ordering trap as compile_shader(source=) in 0.16.
        .def(
            "load_image",
            [](Context& self, const py::bytes& blob, bool mipmaps, const std::string& name) -> py::object
            {
                auto* manager = static_cast<UploadManager*>(self.upload_manager());
                const std::string_view view = blob;
                std::vector<std::byte> bytes(view.size());
                std::memcpy(bytes.data(), view.data(), view.size());
                auto image = unwrap(manager->load_memory(std::move(bytes), mipmaps), self.logger().get());
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                return py::cast(image);
            },
            py::arg("data"),
            py::kw_only(),
            py::arg("mipmaps") = true,
            py::arg("name") = "")
        .def(
            "load_image",
            [](Context& self, const std::string& path, bool mipmaps, const std::string& name) -> py::object
            {
                // sRGB with a full mip chain by default: files are pictures. Arrays go
                // through create_image and stay UNORM (arrays are data). `mipmaps` can
                // turn the chain off (e.g. a UI sprite sampled 1:1 wants no minified
                // levels).
                //
                // Returns IMMEDIATELY: the header is validated here (missing or
                // mangled files fail at the call site, and width/height are right
                // away correct), the decode + copy run on the upload worker. The
                // image is usable for recording at once; residency is enforced at
                // submit. img.ready / img.wait() / ctx.wait() are the
                // explicit-control verbs.
                auto* manager = static_cast<UploadManager*>(self.upload_manager());
                auto image = unwrap(manager->load(path, mipmaps), self.logger().get());
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                if (auto* hr = self.hot_reload())
                    hr->watch_image(image, path);
                return py::cast(image);
            },
            py::arg("path"),
            py::kw_only(),
            py::arg("mipmaps") = true,
            py::arg("name") = "")
        // A list of paths → a layered image loaded from files: texture array
        // (cube=False) or cubemap (cube=True, 6 square faces, order
        // +X,-X,+Y,-Y,+Z,-Z). Async like the single-file load, one batch unit.
        // Hot reload is not wired for layered images (v1): a re-saved face keeps
        // the loaded contents.
        .def(
            "load_image",
            [](Context& self, const std::vector<std::string>& paths, bool cube, bool mipmaps, const std::string& name)
                -> py::object
            {
                auto* manager = static_cast<UploadManager*>(self.upload_manager());
                auto image = unwrap(manager->load_layered(paths, cube, mipmaps), self.logger().get());
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                return py::cast(image);
            },
            py::arg("paths"),
            py::kw_only(),
            py::arg("cube") = false,
            py::arg("mipmaps") = true,
            py::arg("name") = "")
        // Progress of the current batch of uploads, 0.0 .. 1.0 (1.0 when idle) —
        // a loading bar without user-side threads. Covers both kinds: the
        // load_image decodes on the worker, and the one-shot copies of
        // create_buffer and create_image(array), which have nothing to decode
        // and join the batch already submitted.
        .def_property_readonly(
            "upload_progress", [](const Context& self) { return self.upload_manager()->upload_progress(); })
        // Registered before the buffer/list overloads so two ints never reach
        // the buffer protocol. Empty image: 2D, a texture array (layers>1), or a
        // cubemap (cube=True → 6 square faces). Filled by rendering into it or by
        // a compute storage image (procedural skyboxes/arrays); the data forms
        // below upload pixels.
        .def(
            "create_image",
            [](Context& self,
               uint32_t width,
               uint32_t height,
               Format format,
               uint32_t depth,
               uint32_t layers,
               bool cube,
               uint32_t mip_levels,
               const std::string& name) -> py::object
            {
                // depth= is the Z extent of a 3D image, and Vulkan gives a
                // volume exactly one array layer — so the combinations are
                // refused here with the fix, before create_empty re-guards.
                if (depth > 1 && (layers != 1 || cube))
                {
                    raise_error(err_resource(
                        "create_image(depth=): a 3D image cannot have layers or be a cubemap. "
                        "Use depth= alone for a volume, or layers=/cube= on a 2D image."));
                }
                if (cube)
                {
                    if (layers != 1 && layers != 6)
                    {
                        raise_error(err_resource(
                            std::format(
                                "create_image(cube=True) implies 6 layers. Drop layers= or pass layers=6. Got "
                                "layers={}",
                                layers)));
                    }
                    if (width != height)
                    {
                        raise_error(err_resource(
                            std::format(
                                "create_image(cube=True): a cubemap needs square faces, got {}x{}", width, height)));
                    }
                    layers = 6;
                }
                // An empty mipped image allocates the chain; the levels start empty,
                // to be filled by rendering / compute into mip 0 then
                // cmd.generate_mipmaps(). Cap at the dimensions' full chain.
                if (width > 0 && height > 0 && depth > 0)
                {
                    const uint32_t max_mips = Image::full_mip_count(width, height, depth);
                    if (mip_levels < 1 || mip_levels > max_mips)
                    {
                        raise_error(err_resource(
                            std::format(
                                "create_image: mip_levels must be 1..{} for a {}x{}x{} image, got {}",
                                max_mips,
                                width,
                                height,
                                depth,
                                mip_levels)));
                    }
                }
                auto image = unwrap(
                    Image::create_empty(
                        self, width, height, format, mip_levels, layers, cube, VK_SAMPLE_COUNT_1_BIT, depth),
                    self.logger().get());
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                return py::cast(image);
            },
            py::arg("width"),
            py::arg("height"),
            py::arg("format") = Format::RGBA8,
            py::kw_only(),
            py::arg("depth") = 1,
            py::arg("layers") = 1,
            py::arg("cube") = false,
            py::arg("mip_levels") = 1,
            py::arg("name") = "")
        // A single (h,w[,c]) array → a 2D image. cube=True here is a mistake: a
        // cubemap needs six faces, so point the caller at the list form.
        .def(
            "create_image",
            [](Context& self, py::buffer b, bool mipmaps, bool cube, const std::string& name) -> py::object
            {
                if (cube)
                {
                    raise_error(err_resource(
                        "create_image(cube=True): a cubemap needs 6 square faces — pass a list of 6 arrays, "
                        "e.g. create_image([px, nx, py, ny, pz, nz], cube=True)"));
                }
                py::buffer_info info = b.request();
                contiguous_nbytes(info, "create_image");
                // A 4-dim (d, h, w, c) array makes a volume; spec.depth carries it.
                const ArrayImageSpec spec = array_image_spec(info);
                auto image = unwrap(
                    Image::create_from_pixels(
                        self, info.ptr, spec.width, spec.height, spec.format, mipmaps, spec.depth),
                    self.logger().get());
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                return py::cast(image);
            },
            py::arg("array"),
            py::kw_only(),
            py::arg("mipmaps") = false,
            py::arg("cube") = false,
            py::arg("name") = "")
        // A list of arrays → a layered image: texture array (cube=False) or
        // cubemap (cube=True, exactly 6 square faces, order +X,-X,+Y,-Y,+Z,-Z).
        // Every layer must share shape and dtype.
        .def(
            "create_image",
            [](Context& self, py::list images, bool mipmaps, bool cube, const std::string& name) -> py::object
            {
                const size_t layers = images.size();
                if (layers == 0)
                {
                    raise_error(err_resource("create_image: the image list is empty"));
                }
                if (cube && layers != 6)
                {
                    raise_error(err_resource(
                        std::format("create_image(cube=True): a cubemap needs exactly 6 faces, got {}", layers)));
                }

                std::vector<py::buffer_info> infos;
                infos.reserve(layers);
                for (auto item : images)
                {
                    infos.push_back(py::cast<py::buffer>(item).request());
                }

                std::optional<ArrayImageSpec> spec;
                for (size_t i = 0; i < layers; ++i)
                {
                    contiguous_nbytes(infos[i], "create_image");
                    const ArrayImageSpec s = array_image_spec(infos[i]);
                    if (s.depth > 1)
                    {
                        raise_error(err_resource(
                            "create_image: a list of volumes has no meaning. A layer is a (h, w[, c]) "
                            "array; a 3D image is ONE (d, h, w, c) array."));
                    }
                    if (!spec)
                    {
                        spec = s;
                    }
                    else if (s.format != spec->format || s.width != spec->width || s.height != spec->height)
                    {
                        raise_error(err_resource(
                            std::format("create_image: every layer must share shape and dtype. Layer {} differs", i)));
                    }
                }
                if (cube && spec->width != spec->height)
                {
                    raise_error(err_resource(
                        std::format(
                            "create_image(cube=True): faces must be square, got {}x{}", spec->width, spec->height)));
                }

                // Concatenate the layers into one contiguous block: a layered
                // buffer→image copy reads them consecutively from offset 0.
                const size_t layer_bytes = static_cast<size_t>(infos[0].size) * infos[0].itemsize;
                std::vector<std::byte> pixels(layer_bytes * layers);
                for (size_t i = 0; i < layers; ++i)
                {
                    std::memcpy(pixels.data() + i * layer_bytes, infos[i].ptr, layer_bytes);
                }

                auto image = unwrap(
                    Image::create_layered_from_pixels(
                        self,
                        pixels.data(),
                        spec->width,
                        spec->height,
                        static_cast<uint32_t>(layers),
                        cube,
                        spec->format,
                        mipmaps),
                    self.logger().get());
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                return py::cast(image);
            },
            py::arg("images"),
            py::kw_only(),
            py::arg("mipmaps") = false,
            py::arg("cube") = false,
            py::arg("name") = "")
        // From an Image on another Context. A fourth overload of create_image
        // rather than a verb of its own: this is still "make an image on this
        // Context", the source is just where the pixels come from — same
        // reasoning that made cubemaps a `cube=` kwarg instead of create_cubemap.
        //
        // Without external memory (waiting room) the only portable route between
        // two devices is host memory, so this is a readback on the source plus an
        // upload here. `source.read()` + create_image(array) does the same thing
        // in Python — what it cannot do is carry the format, the layer count and
        // the cube-ness across, because a numpy array has nowhere to put them.
        .def(
            "create_image",
            [](Context& self, std::shared_ptr<Image> source, std::string name) -> py::object
            {
                const bool mipmaps = source->mip_levels() > 1;
                const std::uint32_t layers = source->array_layers();
                std::expected<std::vector<std::byte>, Error> bytes;
                {
                    // Blocking on the SOURCE's queue: it stalls that GPU, not this
                    // one. A setup operation, never a per-frame one. The failure
                    // travels back as data and is unwrapped once the GIL is back —
                    // raising under a released GIL is the 0.14 access violation.
                    py::gil_scoped_release release;
                    bytes = source->read(/*all_layers=*/true);
                }
                std::vector<std::byte> pixels = unwrap(std::move(bytes), self.logger().get());

                auto image = layers > 1 ? unwrap(
                                              Image::create_layered_from_pixels(
                                                  self,
                                                  pixels.data(),
                                                  source->width(),
                                                  source->height(),
                                                  layers,
                                                  source->is_cube(),
                                                  source->format(),
                                                  mipmaps),
                                              self.logger().get())
                                        : unwrap(
                                              Image::create_from_pixels(
                                                  self,
                                                  pixels.data(),
                                                  source->width(),
                                                  source->height(),
                                                  source->format(),
                                                  mipmaps,
                                                  source->depth()),
                                              self.logger().get());

                // Levels above 0 arrive from the SOURCE rather than being
                // regenerated here. 0.15 shipped the regenerating version and
                // recorded the loss as a ceiling: "a hand-authored mip chain
                // flattens into a generated one", which is a silent, plausible
                // wrong answer for anyone who rendered their own levels (a
                // roughness-prefiltered environment map is exactly that).
                //
                // One readback per (layer, mip) and one update each. Slow, and
                // deliberately so: the whole overload is already documented as a
                // setup step that blocks the source queue, and correct data
                // beats fast wrong data at setup time.
                const std::uint32_t shared_mips = (std::ranges::min)(source->mip_levels(), image->mip_levels());
                if (shared_mips > 1)
                {
                    auto* manager = static_cast<UploadManager*>(self.upload_manager());
                    for (std::uint32_t mip = 1; mip < shared_mips; ++mip)
                    {
                        const std::uint32_t w = mip_extent(source->width(), mip);
                        const std::uint32_t h = mip_extent(source->height(), mip);
                        const std::uint32_t d = mip_extent(source->depth(), mip);
                        for (std::uint32_t layer = 0; layer < layers; ++layer)
                        {
                            std::expected<std::vector<std::byte>, Error> level;
                            {
                                py::gil_scoped_release release;
                                level = source->read(/*all_layers=*/false, layer, mip);
                            }
                            manager->update(
                                image,
                                unwrap(std::move(level), self.logger().get()),
                                layer,
                                mip,
                                VkOffset3D{0, 0, 0},
                                VkExtent3D{w, h, d});
                        }
                    }
                }
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                return py::cast(image);
            },
            py::arg("source"),
            py::kw_only(),
            py::arg("name") = "")
        // The same two bodies as the RenderTarget constructor, spelled from the
        // Context (0.23): everything else the Context creates comes from a
        // create_* verb, and the target was one of two stragglers — remembering
        // which convention each type uses was a coin flip. The class stays; both
        // spellings share one helper, so they cannot drift.
        .def(
            "create_render_target",
            [](Context& self,
               std::uint32_t width,
               std::uint32_t height,
               py::object color,
               py::object depth,
               std::uint32_t samples,
               std::uint32_t layers,
               bool cube,
               std::uint32_t mip_levels,
               const std::string& name)
            {
                return make_offscreen_target(
                    self, width, height, color, depth, samples, layers, cube, mip_levels, name);
            },
            py::arg("width"),
            py::arg("height"),
            py::arg("color") = Format::RGBA8,
            py::arg("depth") = py::none(),
            py::arg("samples") = 1,
            py::kw_only(),
            py::arg("layers") = 1,
            py::arg("cube") = false,
            py::arg("mip_levels") = 1,
            py::arg("name") = "")
        .def(
            "create_render_target",
            [](Context& self, py::object color, py::object depth, std::uint32_t samples, const std::string& name)
            { return make_offscreen_target_from_images(self, color, depth, samples, name); },
            py::kw_only(),
            py::arg("color") = py::none(),
            py::arg("depth") = py::none(),
            py::arg("samples") = 1,
            py::arg("name") = "")
        .def(
            "create_sampler",
            [](Context& self,
               Filter filter,
               AddressMode address_mode,
               bool anisotropy,
               std::optional<CompareOp> compare,
               BorderColor border_color,
               float mip_lod_bias,
               const std::string& name) -> py::object
            {
                // Cached: identical descriptions return the identical object.
                // Which is why name= accumulates rather than replacing — see
                // Sampler::add_debug_name.
                return py::cast(unwrap(
                    self.get_sampler(
                        SamplerDesc{filter, address_mode, anisotropy, compare, border_color, mip_lod_bias}, name),
                    self.logger().get()));
            },
            py::arg("filter") = Filter::LINEAR,
            py::arg("address_mode") = AddressMode::REPEAT,
            py::arg("anisotropy") = true,
            py::arg("compare") = py::none(),
            py::arg("border_color") = BorderColor::OPAQUE_BLACK,
            py::arg("mip_lod_bias") = 0.0f,
            py::arg("name") = "")
        .def(
            "create_descriptor_pool",
            [](Context& self,
               std::optional<uint32_t> maxSets,
               std::optional<uint32_t> textures,
               std::optional<uint32_t> uniformBuffers,
               std::optional<uint32_t> storageBuffers,
               std::optional<uint32_t> storageImages) -> py::object
            {
                // No sizes at all is the automatic pool (0.23): it grows a block
                // whenever one fills, each sized from the layout being served,
                // so the caller stops doing arithmetic that depended on
                // frames_in_flight — a number the old call never mentioned. Any
                // explicit size is the fixed single pool, the escape hatch.
                if (!maxSets && !textures && !uniformBuffers && !storageBuffers && !storageImages)
                {
                    return py::cast(unwrap(DescriptorPool::create_auto(self), self.logger().get()));
                }
                if (!maxSets)
                {
                    // Sized pools always named max_sets; a half-specified pool
                    // has no sensible reading.
                    throw py::value_error(
                        "create_descriptor_pool: pass max_sets together with the descriptor counts "
                        "for a fixed-size pool, or no arguments at all for the automatic one");
                }
                return py::cast(unwrap(
                    DescriptorPool::create(
                        self,
                        *maxSets,
                        textures.value_or(0),
                        uniformBuffers.value_or(0),
                        storageBuffers.value_or(0),
                        storageImages.value_or(0)),
                    self.logger().get()));
            },
            py::arg("max_sets") = py::none(),
            // textures=, not samplers=: the builder declarator is .texture()
            // and all three name VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER —
            // one name per thing (0.23, breaking).
            py::arg("textures") = py::none(),
            py::arg("uniform_buffers") = py::none(),
            py::arg("storage_buffers") = py::none(),
            py::arg("storage_images") = py::none())
        // Command buffers come from the Context, not a renderer: they are a device
        // resource, and a headless Context has no renderer to ask.
        .def(
            "create_command_buffer",
            [](Context& self, std::optional<bool> auto_barriers) -> py::object
            { return py::cast(unwrap(CommandBuffer::create(self, auto_barriers), self.logger().get())); },
            py::arg("auto_barriers") = py::none())
        // The headless counterpart of renderer.present(): no swapchain, no present.
        .def(
            "submit",
            [](Context& self, std::shared_ptr<CommandBuffer> cmd, bool wait)
            {
                std::expected<void, Error> r;
                {
                    // May block (wait-idle inside when wait=True, and the ring
                    // slot wait either way) — release the GIL for the duration.
                    py::gil_scoped_release release;
                    r = context_submit(self, std::move(cmd), wait);
                }
                unwrap(std::move(r), self.logger().get());
            },
            py::arg("cmd"),
            py::kw_only(),
            py::arg("wait") = true)
        // The one wait verb: every upload and every submit this Context started,
        // finished. The other half of submit(wait=False), and where deferred
        // destruction is reclaimed for that work. Waits on the submission
        // timeline rather than the device, so the other Contexts sharing the
        // device are unaffected.
        .def(
            "wait",
            [](Context& self)
            {
                std::expected<void, Error> r;
                {
                    py::gil_scoped_release release;
                    r = self.wait_for_submits();
                }
                unwrap(std::move(r), self.logger().get());
            });
}
