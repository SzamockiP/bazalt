#include "Bindings.hpp"

void bind_resources(py::module_& m)
{
    py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer")
        // offset= is a byte offset into the buffer. Without it, changing one
        // matrix of an instance array rewrites the whole array — the update is
        // already the per-frame path, so the wasted copy is per frame too.
        .def(
            "update",
            [](Buffer& buffer, std::string_view data, size_t offset)
            {
                require_open(buffer.owner(), "Buffer.update");
                unwrap(buffer.update(std::as_bytes(std::span(data.data(), data.size())), offset), nullptr);
            },
            py::arg("data"),
            py::kw_only(),
            py::arg("offset") = 0)
        .def(
            "update",
            [](Buffer& buffer, py::buffer b, size_t offset)
            {
                require_open(buffer.owner(), "Buffer.update");
                py::buffer_info info = b.request();
                const size_t nbytes = contiguous_nbytes(info, "Buffer.update");
                unwrap(buffer.update({static_cast<const std::byte*>(info.ptr), nbytes}, offset), nullptr);
            },
            py::arg("data"),
            py::kw_only(),
            py::arg("offset") = 0)
        .def(
            "update",
            [](Buffer& buffer, py::list list, std::optional<DataType> dataType, size_t offset)
            {
                require_open(buffer.owner(), "Buffer.update");
                if (list.empty())
                    return;
                DataType actualType = resolve_data_type(list, dataType, DataType::INT32);
                with_list_bytes(
                    list,
                    actualType,
                    [&](const void* data, size_t nbytes)
                    { unwrap(buffer.update({static_cast<const std::byte*>(data), nbytes}, offset), nullptr); });
            },
            py::arg("data"),
            py::arg("data_type") = py::none(),
            py::kw_only(),
            py::arg("offset") = 0)
        // dtype is mandatory: buffers carry no format (unlike Images), so the
        // caller has to say how to interpret the bytes.
        .def(
            "read",
            [](Buffer& self, py::object dtype) -> py::array
            {
                require_open(self.owner(), "Buffer.read");
                auto bytes = unwrap(self.read_bytes(), nullptr);
                const py::dtype dt = py::dtype::from_args(dtype);
                const auto itemsize = static_cast<size_t>(dt.itemsize());
                if (itemsize == 0 || bytes.size() % itemsize != 0)
                {
                    raise_error(err_resource(
                        std::format(
                            "Buffer.read: buffer size {} is not a multiple of the dtype's "
                            "item size {}",
                            bytes.size(),
                            itemsize)));
                }
                py::array out(dt, static_cast<py::ssize_t>(bytes.size() / itemsize));
                std::memcpy(out.mutable_data(), bytes.data(), bytes.size());
                return out;
            },
            py::arg("dtype"))
        // The same pair Image carries, for the same reason: create_buffer does
        // not wait for its staging copy, so the buffer is its own future. You
        // never have to ask — a submit that binds it waits GPU-side and read()
        // waits CPU-side — which leaves these for loading screens and for
        // timing a setup phase.
        .def_property_readonly("ready", &Buffer::ready)
        .def(
            "wait",
            [](Buffer& self)
            {
                require_open(self.owner(), "Buffer.wait");
                py::gil_scoped_release release;
                self.wait();
            });

    py::class_<ShaderModule, std::shared_ptr<ShaderModule>>(m, "ShaderModule")
        .def_property_readonly("path", &ShaderModule::path)
        .def_property_readonly("includes", &ShaderModule::includes)
        .def_property_readonly(
            "spirv",
            [](const ShaderModule& self)
            {
                const auto& words = self.spirv();
                return py::bytes(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(uint32_t));
            })
        // Which (set, binding) pairs this shader writes, from SPIR-V reflection.
        //
        // Bound rather than kept internal for one reason: it is the only way the
        // parser gets a referee in CI. Sync validation is skipped there (debt #4),
        // so without this property the atomics path, the access-chain path and the
        // fail-open cases are checked by nothing that runs on a runner. It is also
        // the answer to "why is there no barrier here".
        .def_property_readonly(
            "writes",
            [](const ShaderModule& self)
            {
                const auto& reflection = self.reflection();
                py::list out;
                for (const auto& [set, binding] : reflection.written_bindings)
                {
                    out.append(py::make_tuple(set, binding));
                }
                return out;
            })
        // True when the write scan could not follow something and every binding is
        // therefore assumed written. See the invariant in SpirvReflect.hpp.
        .def_property_readonly(
            "writes_unknown", [](const ShaderModule& self) { return self.reflection().writes_unknown; })
        .def_property_readonly("prints", [](const ShaderModule& self) { return self.reflection().prints; });

    // Image + Sampler replace the old Texture, which fused VkImage, view and a
    // per-texture sampler into one object. Samplers are cached on the Context;
    // an Image is just the pixels.
    py::class_<Image, std::shared_ptr<Image>>(m, "Image")
        .def_property_readonly("width", &Image::width)
        .def_property_readonly("height", &Image::height)
        .def_property_readonly("depth", &Image::depth)
        .def_property_readonly("format", &Image::format)
        .def_property_readonly("mip_levels", &Image::mip_levels)
        .def_property_readonly("array_layers", &Image::array_layers)
        .def_property_readonly("is_cube", &Image::is_cube)
        .def_property_readonly("samples", &Image::samples)
        .def_property_readonly("ready", &Image::ready)
        .def(
            "wait",
            [](Image& self)
            {
                require_open(self.owner(), "Image.wait");
                std::expected<void, Error> r;
                {
                    py::gil_scoped_release release;
                    r = self.wait();
                }
                unwrap(std::move(r), nullptr);
            })
        .def(
            "read",
            [](Image& self, std::uint32_t layer, std::uint32_t mip) -> py::array
            {
                require_open(self.owner(), "Image.read");
                return image_to_numpy(self, layer, mip);
            },
            py::kw_only(),
            py::arg("layer") = 0,
            py::arg("mip") = 0)
        // Change the pixels of an image that already exists. See the stub for
        // the why; the work here is deciding every user error on the main
        // thread, because the worker cannot raise.
        .def(
            "update",
            [](std::shared_ptr<Image> self,
               const py::array& array,
               std::uint32_t layer,
               std::uint32_t mip,
               const py::object& region)
            {
                // ResourceError, not ValueError: every one of these needs the image
                // to decide, and image.read() answers the same questions the same
                // way. See "Which exception a user error gets" in DESIGN.md.
                Context* context = const_cast<Context*>(self->owner());
                if (!context)
                {
                    raise_error(err_resource("update(): this image has no Context"));
                }
                require_open(*context, "Image.update");
                if (self->samples() != 1)
                {
                    raise_error(err_resource(
                        "update(): a multisampled image cannot be uploaded to. It is rendered "
                        "into and resolved out"));
                }
                if (layer >= self->array_layers())
                {
                    raise_error(err_resource(
                        std::format("update(layer={}): this image has {} layer(s)", layer, self->array_layers())));
                }
                if (mip >= self->mip_levels())
                {
                    raise_error(err_resource(
                        std::format("update(mip={}): this image has {} mip level(s)", mip, self->mip_levels())));
                }

                const std::uint32_t level_w = mip_extent(self->width(), mip);
                const std::uint32_t level_h = mip_extent(self->height(), mip);
                const std::uint32_t level_d = mip_extent(self->depth(), mip);
                std::uint32_t x = 0, y = 0, z = 0, w = level_w, h = level_h, d = level_d;
                if (!region.is_none())
                {
                    py::sequence seq = py::cast<py::sequence>(region);
                    // ValueError, unlike the rest of them: a wrong-length tuple is
                    // malformed on its own, and no image has to be consulted to say
                    // so. A 2D image takes 4 numbers, a volume takes 6 — the
                    // dimension count of the image decides which, exactly as the
                    // array shape does on create_image.
                    const bool volume = self->is_3d();
                    const size_t expected_len = volume ? 6 : 4;
                    if (py::len(seq) != expected_len)
                    {
                        throw py::value_error(
                            volume ? "update(region=): a 3D image expects (x, y, z, width, height, depth)"
                                   : "update(region=): expected (x, y, width, height)");
                    }
                    if (volume)
                    {
                        x = py::cast<std::uint32_t>(seq[0]);
                        y = py::cast<std::uint32_t>(seq[1]);
                        z = py::cast<std::uint32_t>(seq[2]);
                        w = py::cast<std::uint32_t>(seq[3]);
                        h = py::cast<std::uint32_t>(seq[4]);
                        d = py::cast<std::uint32_t>(seq[5]);
                    }
                    else
                    {
                        x = py::cast<std::uint32_t>(seq[0]);
                        y = py::cast<std::uint32_t>(seq[1]);
                        w = py::cast<std::uint32_t>(seq[2]);
                        h = py::cast<std::uint32_t>(seq[3]);
                        d = 1;
                    }
                    if (w == 0 || h == 0 || d == 0 || !fits_within(x, w, level_w) || !fits_within(y, h, level_h) ||
                        !fits_within(z, d, level_d))
                    {
                        raise_error(err_resource(
                            std::format(
                                "update(region=): the {}x{}x{} region at ({}, {}, {}) does not fit in the "
                                "{}x{}x{} of mip {}",
                                w,
                                h,
                                d,
                                x,
                                y,
                                z,
                                level_w,
                                level_h,
                                level_d,
                                mip)));
                    }
                }

                std::vector<std::byte> pixels = update_pixels_from_numpy(*self, array, w, h, d);

                auto* manager = static_cast<UploadManager*>(context->upload_manager());
                manager->update(
                    std::move(self),
                    std::move(pixels),
                    layer,
                    mip,
                    VkOffset3D{
                        static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), static_cast<std::int32_t>(z)},
                    VkExtent3D{w, h, d});
            },
            py::arg("array"),
            py::kw_only(),
            py::arg("layer") = 0,
            py::arg("mip") = 0,
            py::arg("region") = py::none());

    // name is readable because it accumulates: the cache shares one sampler
    // between identical descriptions, so what the object is called is the list
    // of everyone who named it, and a caller cannot predict that from its own
    // create_sampler call alone.
    py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Sampler").def_property_readonly("name", &Sampler::debug_name);
}
