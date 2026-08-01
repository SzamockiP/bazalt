#pragma once
// Everything the binding translation units share.
//
// This was the anonymous namespace and the file-scope statics of main.cpp until
// 0.20. It is a header now because the bindings live in eight .cpp files, and it
// is `inline` rather than anonymous for a reason worth stating: an anonymous
// namespace here would give every translation unit its OWN copy of the exc_*
// handles, register_exceptions would fill in main.cpp's copy alone, and every
// raise_error from any other file would go through a null handle.
//
// The include list is deliberately the whole of it rather than a per-file
// minimum. A missing include is a build break the next person has to diagnose,
// the precompiled header pays the cost once, and nothing here is worth the
// bookkeeping.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <span>
#include <chrono>

#include "Error.hpp"
#include "Logger.hpp"
#include "window.hpp"
#include "Context.hpp"
#include "RenderTarget.hpp"
#include "Renderer.hpp"
#include "Buffer.hpp"
#include "ShaderCompiler.hpp"
#include "Pipeline.hpp"
#include "CommandBuffer.hpp"
#include "Format.hpp"
#include "Image.hpp"
#include "Sampler.hpp"
#include "UploadManager.hpp"
#include "HotReload.hpp"
#include "DescriptorSet.hpp"

namespace py = pybind11;

// ── Error boundary ────────────────────────────────────────────────────────────
//
// Every ErrorCode gets its own Python class so that recoverability is expressible
// as `except bz.ShaderError`. Previously every failure — a shader typo and a lost
// device alike — arrived as a bare RuntimeError, so callers had no way to keep
// running after the recoverable ones.

inline py::handle exc_bazalt;
inline py::handle exc_initialization;
inline py::handle exc_device_lost;
inline py::handle exc_out_of_memory;
inline py::handle exc_shader;
inline py::handle exc_window;
inline py::handle exc_resource;
inline py::handle exc_state;
inline py::handle exc_unsupported;

inline py::handle make_exception(py::module_& m, const char* name, py::handle base)
{
    std::string qualified = std::string("bazalt._core.") + name;
    py::object exc = py::reinterpret_steal<py::object>(PyErr_NewException(qualified.c_str(), base.ptr(), nullptr));
    m.add_object(name, exc);
    return exc.release();
}

inline void register_exceptions(py::module_& m)
{
    exc_bazalt = make_exception(m, "BazaltError", PyExc_Exception);
    exc_initialization = make_exception(m, "InitializationError", exc_bazalt);
    exc_device_lost = make_exception(m, "DeviceLostError", exc_bazalt);
    exc_out_of_memory = make_exception(m, "OutOfMemoryError", exc_bazalt);
    exc_shader = make_exception(m, "ShaderError", exc_bazalt);
    exc_window = make_exception(m, "WindowError", exc_bazalt);
    exc_resource = make_exception(m, "ResourceError", exc_bazalt);
    exc_state = make_exception(m, "StateError", exc_bazalt);
    exc_unsupported = make_exception(m, "UnsupportedError", exc_bazalt);
}

inline py::handle exception_for(ErrorCode code)
{
    switch (code)
    {
        case ErrorCode::Initialization:
            return exc_initialization;
        case ErrorCode::DeviceLost:
            return exc_device_lost;
        case ErrorCode::OutOfMemory:
            return exc_out_of_memory;
        case ErrorCode::Shader:
            return exc_shader;
        case ErrorCode::Window:
            return exc_window;
        case ErrorCode::Resource:
            return exc_resource;
        case ErrorCode::State:
            return exc_state;
        case ErrorCode::Unsupported:
            return exc_unsupported;
    }
    return exc_bazalt;
}

[[noreturn]] inline void raise_error(const Error& error)
{
    py::handle type = exception_for(error.code);
    py::object instance =
        py::reinterpret_steal<py::object>(PyObject_CallFunction(type.ptr(), "s", error.message.c_str()));

    // Diagnostics travel as attributes rather than being smashed into the text,
    // so tooling can branch on them.
    if (error.result != VK_SUCCESS)
    {
        instance.attr("vk_result") = std::string(vk_result_name(error.result));
    }
    if (error.code == ErrorCode::Shader)
    {
        instance.attr("path") = error.path;
        instance.attr("line") = error.line;
    }

    PyErr_SetObject(type.ptr(), instance.ptr());
    throw py::error_already_set();
}

// Collapses the log-then-throw block that was copy-pasted at every call site.
template <typename T>
T unwrap(std::expected<T, Error>&& result, Logger* logger)
{
    if (result)
    {
        return std::move(result.value());
    }
    if (logger)
    {
        logger->log(result.error());
    }
    raise_error(result.error());
}

inline void unwrap(std::expected<void, Error>&& result, Logger* logger)
{
    if (result)
    {
        return;
    }
    if (logger)
    {
        logger->log(result.error());
    }
    raise_error(result.error());
}

// Two Contexts can be alive since 0.15, so "a resource from the other
// Context" is now a mistake a user can actually make — and one Vulkan
// punishes with a driver crash or an unattributed validation message rather
// than an exception. Every place a foreign object could first enter a
// recording asks this instead. Costs a pointer comparison at record time.
//
// It lives at the binding layer, not in the headers: this guards a user
// error, and the recording methods return `*this` for chaining, so giving
// them an error channel would be a rebuild for one check. The GIL is held
// here, which is what makes raising legal at all (see the 0.14 lesson about
// raise_error under gil_scoped_release).
inline void require_same_context(const Context* a, const Context* b, const char* what)
{
    if (a != nullptr && b != nullptr && a != b)
    {
        raise_error(err_resource(
            std::format(
                "{}: that object belongs to a different Context. Resources cannot "
                "cross Contexts — build them from the Context you are recording "
                "for, or copy the data over (ctx.create_image(image_from_the_other_context)).",
                what)));
    }
}

// clear_color=None preserves the attachment, and a multisampled target has
// nothing to preserve: its multisampled image is transient (storeOp
// DONT_CARE) and the pass result lives in the resolve image, which is not
// what the next pass renders into. Lifting the ceiling means storing the
// multisampled image, which costs every MSAA pass to serve the rare one.
//
// Here rather than in CommandBuffer for the same reason as
// require_same_context above: a user error, and the recording methods chain.
inline void require_preservable(const RenderTarget& target, bool preserve, const char* what)
{
    if (preserve && target.samples() != VK_SAMPLE_COUNT_1_BIT)
    {
        raise_error(err_resource(
            std::format(
                "{}(clear_color=None) preserves the attachment, but a multisampled target has "
                "nothing to preserve. The multisampled image is discarded at the end of each pass. "
                "The result goes to the resolve image, and the next pass does not render into that "
                "image. Use a target with samples=1 for a multi-pass sequence.",
                what)));
    }
}

// `blend()`'s two spellings resolved into the one equation the pipeline stores
// (0.23). A named mode is four points in the factor space; the factors are the
// axes, for the fifth thing somebody wants (rule 2: leave the hatch).
//
// Three refusals, all ValueError, because each is malformed on its own — no
// resource, no data and no device enters the decision, which is the 0.20 line.
// The alternative to refusing is a silent winner between two ways of saying one
// thing, and "which one wins" is a rule nobody remembers at 3am.
inline BlendEquation resolve_blend_equation(
    const std::optional<BlendMode>& mode,
    const std::optional<BlendFactor>& src,
    const std::optional<BlendFactor>& dst,
    const std::optional<BlendOp>& op,
    const std::optional<BlendFactor>& src_alpha,
    const std::optional<BlendFactor>& dst_alpha,
    const std::optional<BlendOp>& alpha_op)
{
    const bool any_factor = src || dst || op || src_alpha || dst_alpha || alpha_op;
    if (mode && any_factor)
    {
        throw py::value_error(
            "blend(): mode= and the factor arguments are two ways to say the same thing. "
            "Pass a mode for a named blend, or src=/dst=/op= for a hand-written one.");
    }
    if (!any_factor)
    {
        // No mode either is the historical default: blend(True) is ALPHA.
        return blend_equation_for(mode.value_or(BlendMode::ALPHA));
    }
    // A factor spelling names a COMPLETE equation, so the colour pair is
    // required together: half of one has no reading that is not a guess about
    // the other half.
    if (!src || !dst)
    {
        throw py::value_error(
            "blend(): src= and dst= go together — they are the two sides of one equation. "
            "Pass both, or a mode= for a named blend.");
    }
    BlendEquation eq{*src, *dst, op.value_or(BlendOp::ADD), *src, *dst, op.value_or(BlendOp::ADD)};
    // The alpha channel follows the colour unless it is spelled out, which is
    // what glBlendFunc does and what nearly every blend wants. Spelled out, the
    // same completeness rule applies to the pair.
    if (src_alpha || dst_alpha || alpha_op)
    {
        if (!src_alpha || !dst_alpha)
        {
            throw py::value_error(
                "blend(): src_alpha= and dst_alpha= go together. Pass both to give alpha its "
                "own equation, or neither to have it follow the colour.");
        }
        eq.src_alpha = *src_alpha;
        eq.dst_alpha = *dst_alpha;
        eq.alpha_op = alpha_op.value_or(BlendOp::ADD);
    }
    return eq;
}

// ── What the Context makes ────────────────────────────────────────────────────
//
// One rule with no exceptions since 0.23: `bz.Context` and `bz.Window` are the
// roots — nothing owns them, so they are constructors — and every resource a
// Context owns comes from a `ctx.create_*` verb. Buffers, images, samplers,
// pools, pipelines and command buffers always did; the render target and the
// swapchain renderer were top-level constructors until 0.23, which made "which
// convention does this type use" a coin flip to memorize.
//
// The constructors are GONE rather than kept as aliases. A second spelling of
// the same call is a fork, not a convenience (the 0.18 audit), and these two
// would have been the seventh and eighth: same work, same arguments, one
// paragraph of documentation explaining which to prefer — which is the tell.
//
// The bodies live here rather than inline in ContextBind.cpp because each is
// sixty lines of argument-shape parsing, and that file is long enough.

// Allocating form: the target creates its attachments from formats.
inline std::shared_ptr<OffscreenTarget> make_offscreen_target(
    Context& context,
    std::uint32_t width,
    std::uint32_t height,
    const py::object& color,
    const py::object& depth,
    std::uint32_t samples,
    std::uint32_t layers,
    bool cube,
    std::uint32_t mip_levels,
    const std::string& name)
{
    std::vector<Format> colors;
    if (!color.is_none())
    {
        if (py::isinstance<Format>(color))
        {
            colors.push_back(color.cast<Format>());
        }
        else if (py::isinstance<py::sequence>(color) && !py::isinstance<py::str>(color))
        {
            for (auto item : color.cast<py::sequence>())
            {
                // Images here mean the caller wants the borrowed-attachment
                // overload but also passed width/height, which that overload
                // does not take. pybind cannot fall through once this signature
                // has matched, so say so rather than letting item.cast<Format>()
                // report a cast error about a type mismatch the caller did not
                // make.
                if (py::isinstance<Image>(item))
                {
                    raise_error(err_resource(
                        "to render into images you already own, drop width and height: "
                        "ctx.create_render_target(color=[image]) — the size, layers and mip "
                        "levels come off the images"));
                }
                colors.push_back(item.cast<Format>());
            }
        }
        else if (py::isinstance<Image>(color))
        {
            raise_error(err_resource(
                "to render into an image you already own, drop width and height: "
                "ctx.create_render_target(color=[image]) — the size, layers and mip "
                "levels come off the images"));
        }
        else
        {
            raise_error(err_resource("color must be a bz.Format, a list of them, or None"));
        }
    }

    std::optional<Format> depth_format;
    if (!depth.is_none())
    {
        if (py::isinstance<py::bool_>(depth))
        {
            raise_error(err_resource(
                "depth is a pixel format now: pass depth=bz.Format.D32F "
                "instead of depth=True"));
        }
        depth_format = depth.cast<Format>();
    }

    return unwrap(
        OffscreenTarget::create(
            context, width, height, std::move(colors), depth_format, samples, layers, cube, mip_levels, name),
        context.logger().get());
}

// Borrowed-images form: the caller's Images become the attachments (with MSAA
// they become the resolve targets).
inline std::shared_ptr<OffscreenTarget> make_offscreen_target_from_images(
    Context& context,
    const py::object& color,
    const py::object& depth,
    std::uint32_t samples,
    const std::string& name)
{
    std::vector<std::shared_ptr<Image>> colors;
    if (!color.is_none())
    {
        if (py::isinstance<Image>(color))
        {
            colors.push_back(color.cast<std::shared_ptr<Image>>());
        }
        else if (py::isinstance<py::sequence>(color) && !py::isinstance<py::str>(color))
        {
            for (auto item : color.cast<py::sequence>())
            {
                if (py::isinstance<Format>(item))
                {
                    raise_error(err_resource(
                        "color has pixel formats but no width and height. Pass the size "
                        "to have the target allocate its attachments — "
                        "ctx.create_render_target(512, 512, color=bz.Format.RGBA8) — or pass "
                        "images from ctx.create_image to render into those."));
                }
                colors.push_back(item.cast<std::shared_ptr<Image>>());
            }
        }
        else
        {
            raise_error(err_resource(
                "color must be an Image, a list of them, or None. To have the target "
                "allocate its own attachments, pass a width and height with a bz.Format."));
        }
    }

    std::shared_ptr<Image> depth_image;
    if (!depth.is_none())
    {
        depth_image = depth.cast<std::shared_ptr<Image>>();
    }

    // The cross-Context guard belongs in the binding layer, as it does for
    // every other resource: this catches a user mistake, not a C++ invariant,
    // and the GIL is held here.
    for (const auto& image : colors)
    {
        require_same_context(&context, image->owner(), "RenderTarget(color=)");
    }
    if (depth_image)
    {
        require_same_context(&context, depth_image->owner(), "RenderTarget(depth=)");
    }

    return unwrap(
        OffscreenTarget::create_from_images(context, std::move(colors), std::move(depth_image), samples, name),
        context.logger().get());
}

// The swapchain renderer over a bazalt Window. The Context is the receiver
// (ctx.create_renderer(window)) rather than the second argument, which is the
// 0.23 rule; the window is what it needs, exactly as an Image is what
// create_render_target(color=[...]) needs.
inline std::shared_ptr<SwapchainRenderer> make_swapchain_renderer(
    std::shared_ptr<Context> context,
    Window& window,
    PresentMode present_mode,
    std::uint32_t samples,
    bool stencil)
{
    auto sp = window.get_surface_provider();
    return std::shared_ptr<SwapchainRenderer>(unwrap(
        SwapchainRenderer::create(context, std::move(sp), present_mode, samples, stencil), context->logger().get()));
}

// The same renderer over a window bazalt did not open: a Qt widget, a wx frame,
// anything that can hand over an HWND. Windows only, and it says so.
inline std::shared_ptr<SwapchainRenderer> make_swapchain_renderer_win32(
    std::shared_ptr<Context> context,
    std::uint64_t hwnd,
    PresentMode present_mode,
    std::uint32_t samples,
    bool stencil)
{
#ifdef _WIN32
    SurfaceProvider sp;
    sp.required_instance_extensions = {"VK_KHR_surface", "VK_KHR_win32_surface"};

    sp.create_surface = [hwnd](VkInstance instance) -> VkSurfaceKHR
    {
        auto pfnCreateWin32Surface =
            (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR");
        if (!pfnCreateWin32Surface)
        {
            return VK_NULL_HANDLE;
        }
        VkWin32SurfaceCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .hinstance = GetModuleHandle(nullptr),
            .hwnd = (HWND)hwnd};
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (pfnCreateWin32Surface(instance, &createInfo, nullptr, &surface) != VK_SUCCESS)
        {
            return VK_NULL_HANDLE;
        }
        return surface;
    };

    sp.get_framebuffer_size = [hwnd]() -> std::pair<int, int>
    {
        RECT rect;
        if (GetClientRect((HWND)hwnd, &rect))
        {
            return {rect.right - rect.left, rect.bottom - rect.top};
        }
        return {0, 0};
    };

    auto last_width = std::make_shared<int>(0);
    auto last_height = std::make_shared<int>(0);

    sp.consume_resize_flag = [hwnd, last_width, last_height]() -> bool
    {
        RECT rect;
        if (GetClientRect((HWND)hwnd, &rect))
        {
            int w = rect.right - rect.left;
            int h = rect.bottom - rect.top;
            if (w != *last_width || h != *last_height)
            {
                *last_width = w;
                *last_height = h;
                return true;
            }
        }
        return false;
    };

    return std::shared_ptr<SwapchainRenderer>(unwrap(
        SwapchainRenderer::create(context, std::move(sp), present_mode, samples, stencil), context->logger().get()));
#else
    (void)context;
    (void)hwnd;
    (void)present_mode;
    (void)samples;
    (void)stencil;
    raise_error(err_window("create_renderer(win32_hwnd=) is only supported on Windows"));
#endif
}

// A 3D target is rendered one Z slice at a time: its whole-target view is a 3D
// view, which Vulkan does not accept as an attachment, so the mistake is
// refused with the fix rather than surfacing as a validation error naming a
// view type. Same address as the two guards above: a user error, and the
// recording methods chain.
inline void require_sliced_when_3d(const RenderTarget& target, const char* what)
{
    if (const auto* offscreen = dynamic_cast<const OffscreenTarget*>(&target); offscreen && offscreen->is_3d())
    {
        raise_error(err_resource(
            std::format(
                "{}: a 3D target is rendered one slice at a time. Use target.layer(z) — "
                "cmd.begin_rendering(target.layer(z)) — instead of the whole target.",
                what)));
    }
}

// Resolves a list's element type from the explicit argument or the first
// element. `int_default` is the caller's policy: create_buffer infers UINT32
// for integers going into an INDEX buffer, update infers INT32 — a deliberate
// difference, not drift.
inline DataType resolve_data_type(const py::list& list, std::optional<DataType> requested, DataType int_default)
{
    if (requested.has_value())
    {
        return requested.value();
    }
    if (py::isinstance<py::float_>(list[0]))
    {
        return DataType::FLOAT;
    }
    if (py::isinstance<py::int_>(list[0]))
    {
        return int_default;
    }
    raise_error(err_resource(
        std::format(
            "Bazalt cannot infer the data type from a list of {}. It reads the first "
            "element, and it recognises bool, int and float. Pass data_type= to say it, "
            "for example data_type=bz.DataType.FLOAT.",
            py::str(py::type::of(list[0])).cast<std::string>())));
}

// Calls fn(data, nbytes) with the list packed as the requested element type.
// This four-way ladder used to be written out twice (Buffer.update and
// Context.create_buffer) and had already diverged once: update lacked UINT16.
template <typename F>
auto with_list_bytes(const py::list& list, DataType type, F&& fn)
{
    const size_t count = list.size();
    switch (type)
    {
        case DataType::FLOAT:
        {
            std::vector<float> data(count);
            for (size_t i = 0; i < count; ++i)
                data[i] = list[i].cast<float>();
            return fn(data.data(), count * sizeof(float));
        }
        case DataType::UINT32:
        {
            std::vector<uint32_t> data(count);
            for (size_t i = 0; i < count; ++i)
                data[i] = list[i].cast<uint32_t>();
            return fn(data.data(), count * sizeof(uint32_t));
        }
        case DataType::UINT16:
        {
            std::vector<uint16_t> data(count);
            for (size_t i = 0; i < count; ++i)
                data[i] = list[i].cast<uint16_t>();
            return fn(data.data(), count * sizeof(uint16_t));
        }
        case DataType::INT32:
        {
            std::vector<int32_t> data(count);
            for (size_t i = 0; i < count; ++i)
                data[i] = list[i].cast<int32_t>();
            return fn(data.data(), count * sizeof(int32_t));
        }
    }
    raise_error(err_resource("Unknown data type"));
}

// True when the buffer's bytes are packed in C order.
//
// Dimensions of extent 1 are skipped: their stride is unconstrained and numpy
// leaves arbitrary values there, so comparing them yields false negatives.
inline bool is_c_contiguous(const py::buffer_info& info)
{
    py::ssize_t expected = info.itemsize;
    for (py::ssize_t i = info.ndim - 1; i >= 0; --i)
    {
        if (info.shape[i] == 1)
        {
            continue;
        }
        if (info.strides[i] != expected)
        {
            return false;
        }
        expected *= info.shape[i];
    }
    return true;
}

// Refuses strided input rather than silently uploading garbage.
//
// Copying instead would be friendlier, but a hidden allocation on every upload
// is exactly the kind of invisible cost this library exists to avoid — so the
// copy stays the caller's explicit decision.
inline size_t contiguous_nbytes(const py::buffer_info& info, const char* what)
{
    if (!is_c_contiguous(info))
    {
        raise_error(err_resource(
            std::format(
                "{} requires a C-contiguous array, got a strided view "
                "(e.g. arr.T or arr[::2]). Pass numpy.ascontiguousarray(arr) instead.",
                what)));
    }
    return static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
}

// One (h, w), (h, w, channels) or (d, h, w, channels) array → its GPU Format +
// dimensions. UNORM, never sRGB (arrays are data, files are pictures). Raises a
// ResourceError on an unsupported dtype/shape. Shared by the single-image and
// layered (texture array / cubemap) create_image paths.
//
// The dimension count IS the disambiguation: ndim 3 stays (h, w, channels) as
// it always was, and a volume is ndim 4 — a single-channel volume is one
// arr[..., None] away. A depth= kwarg on the array overload would be a
// validation-only argument, which is the shape the design notes condemn.
struct ArrayImageSpec
{
    Format format;
    uint32_t width;
    uint32_t height;
    uint32_t depth = 1;
};
inline ArrayImageSpec array_image_spec(const py::buffer_info& info)
{
    if (info.ndim != 2 && info.ndim != 3 && info.ndim != 4)
    {
        raise_error(err_resource(
            std::format(
                "create_image expects a (h, w) or (h, w, channels) array — or (d, h, w, "
                "channels) for a 3D image — got {} dimensions",
                info.ndim)));
    }
    const bool volume = info.ndim == 4;
    const auto depth = volume ? static_cast<uint32_t>(info.shape[0]) : 1u;
    const auto height = static_cast<uint32_t>(info.shape[volume ? 1 : 0]);
    const auto width = static_cast<uint32_t>(info.shape[volume ? 2 : 1]);
    const py::ssize_t channels = volume ? info.shape[3] : (info.ndim == 3 ? info.shape[2] : 1);

    std::optional<Format> format;
    if (info.format == "B")
    { // uint8
        if (channels == 1)
            format = Format::R8;
        else if (channels == 2)
            format = Format::RG8;
        else if (channels == 4)
            format = Format::RGBA8;
    }
    else if (info.format == "e")
    { // float16
        if (channels == 1)
            format = Format::R16F;
        else if (channels == 4)
            format = Format::RGBA16F;
    }
    else if (info.format == "f")
    { // float32
        if (channels == 1)
            format = Format::R32F;
        else if (channels == 4)
            format = Format::RGBA32F;
    }

    if (!format)
    {
        if (channels == 3)
        {
            raise_error(err_resource(
                "create_image: 3-channel images have no portable GPU format. "
                "Pad to 4 channels first, e.g. "
                "np.concatenate([arr, np.full_like(arr[..., :1], 255)], axis=-1)"));
        }
        // A raw (d, h, w) float volume lands here with its whole depth read as
        // channels, so the message must teach the 4-dim rule or the caller
        // cannot find it.
        raise_error(err_resource(
            std::format(
                "create_image: unsupported dtype/shape (dtype '{}', {} channel(s)). "
                "Supported: uint8 x 1/2/4 channels, float16 x 1/4, float32 x 1/4. "
                "For a 3D image pass (depth, h, w, channels) — a single-channel "
                "volume is arr[..., None].",
                info.format,
                channels)));
    }
    return {*format, width, height, depth};
}

inline const char* severity_name(Severity severity)
{
    switch (severity)
    {
        case Severity::Info:
            return "INFO";
        case Severity::Warning:
            return "WARNING";
        case Severity::Error:
            return "ERROR";
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints, so a forged
    // Severity from Python must degrade gracefully, not invoke UB.
    return "INFO";
}

inline ValidationMode parse_validation(const std::string& value)
{
    if (value == "auto")
        return ValidationMode::Auto;
    if (value == "on")
        return ValidationMode::On;
    if (value == "off")
        return ValidationMode::Off;
    if (value == "sync")
        return ValidationMode::Sync;
    // ValueError: a name outside a fixed set, and nothing about the device or a
    // resource enters the decision. py::value_error rather than
    // std::invalid_argument, which pybind translates to the same class — one
    // spelling for one outcome.
    throw py::value_error(std::format("validation must be one of 'auto', 'on', 'off', 'sync' (got '{}')", value));
}

// A Context built without a logger used to render with validation off and say
// nothing about its own failures. Default to reporting warnings on stderr.
inline std::shared_ptr<Logger> make_default_logger()
{
    auto logger = std::make_shared<Logger>(Severity::Warning);
    logger->register_callback(
        py::cpp_function(
            [](const LogMessage& msg)
            {
                py::object stderr_stream = py::module_::import("sys").attr("stderr");
                stderr_stream.attr("write")(std::format("[bazalt] {}: {}\n", severity_name(msg.severity), msg.text));
            }));
    return logger;
}

// A timestamp query pair to wrap the frame with (GPU timing). The swapchain
// path passes its pool; the headless path leaves it null and records nothing.
struct TimestampRange
{
    VkQueryPool pool = VK_NULL_HANDLE;
    std::uint32_t first = 0;
};

// Reset, begin, replay and end the per-frame VkCommandBuffer. Shared by the
// swapchain and headless submit paths, which only differ in what happens to
// the recorded buffer afterwards.
// `capture_into` is the renderer that wants a copy of its presentable image
// this frame, or null. It records inside this command buffer because that is
// the only place the image is legally ours (see SwapchainRenderer::read_pixels).
//
// Returns the Error rather than raising it. Both callers reach here from a
// binding that released the GIL, and raise_error without the GIL is an access
// violation, not an exception -- the same 0.14 lesson require_same_context and
// present_command_buffer already carry in their comments. These two paths were
// the last that still raised from under the release: a vkBeginCommandBuffer or
// vkEndCommandBuffer that returns DEVICE_LOST or an out-of-memory result would
// have crashed the interpreter instead of raising bz.DeviceLostError.
inline std::expected<VkCommandBuffer, Error> record_frame(
    CommandBuffer& cmd,
    const Context& ctx,
    TimestampRange ts = {},
    SwapchainRenderer* capture_into = nullptr)
{
    const VolkDeviceTable& vk = ctx.vk();
    const std::uint32_t frame_index = ctx.frame_index();
    VkCommandBuffer vkCmd = cmd.get(frame_index);
    vk.vkResetCommandBuffer(vkCmd, 0);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr};

    if (auto e = check(vk.vkBeginCommandBuffer(vkCmd, &beginInfo), "begin recording command buffer"))
    {
        return std::unexpected(*e);
    }

    // The queries must be reset on the device before use; doing it here (rather
    // than once up front) keeps them per-frame and needs no hostQueryReset.
    if (ts.pool != VK_NULL_HANDLE)
    {
        vk.vkCmdResetQueryPool(vkCmd, ts.pool, ts.first, 2);
        vk.vkCmdWriteTimestamp(vkCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts.pool, ts.first);
    }

    cmd.execute(vkCmd, FrameContext{frame_index, &vk});

    // After the recording, so the copy sees the finished frame; before the
    // closing timestamp, so the capture's cost is visible in gpu_time_ms.
    if (capture_into)
    {
        capture_into->record_capture(vkCmd);
    }

    if (ts.pool != VK_NULL_HANDLE)
    {
        vk.vkCmdWriteTimestamp(vkCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts.pool, ts.first + 1);
    }

    if (auto e = check(vk.vkEndCommandBuffer(vkCmd), "record command buffer"))
    {
        return std::unexpected(*e);
    }

    return vkCmd;
}

inline std::expected<void, Error> SwapchainRenderer::present(
    std::shared_ptr<CommandBuffer> cmd,
    std::uint64_t upload_wait_serial,
    bool capture)
{
    TimestampRange ts{};
    if (timestamps_supported())
    {
        ts = {timestamp_pool(), 2 * current_frame()};
    }
    auto vkCmd = record_frame(*cmd, *context(), ts, capture ? this : nullptr);
    if (!vkCmd)
    {
        // Nothing was submitted, and acquire() already reset this slot's fence.
        // Without this the raise below is followed by a hang on the next frame.
        abandon_frame_();
        return std::unexpected(vkCmd.error());
    }
    end_frame(*vkCmd, upload_wait_serial);
    if (timestamps_supported())
    {
        // The slot now holds results acquire() can read once its fence signals.
        mark_timestamp_written();
    }
    return {};
}

// Upload residency, per command buffer: every image this recording references
// must at least be *submitted* (CPU-wait while the worker is still decoding —
// correctness first), and the frame's GPU work then waits on the submission
// timeline for the highest upload it depends on (zero CPU stall in the steady
// state). Resources with no pending upload — RTT attachments, DYNAMIC
// buffers — short-circuit to 0.
//
// Buffers joined images in 0.18.0, when create_buffer stopped waiting for its
// own staging copy. A buffer has no decode, so it needs no CPU-side half: the
// serial is known the moment create_buffer returns, and one timeline wait
// covers both kinds of upload.
inline std::expected<std::uint64_t, Error> require_uploads_resident(CommandBuffer& cmd)
{
    std::uint64_t wait_serial = 0;
    for (const auto& set : cmd.used_sets())
    {
        for (const auto& bi : set->images())
        {
            auto serial = bi.image->require_resident();
            if (!serial)
            {
                return std::unexpected(serial.error());
            }
            wait_serial = (std::max)(wait_serial, *serial);
        }
        for (const auto& bb : set->buffers())
        {
            wait_serial = (std::max)(wait_serial, bb.buffer->upload_serial());
        }
    }
    // Vertex, index and transfer uses: bound directly rather than through a set.
    for (const auto& buffer : cmd.used_buffers())
    {
        wait_serial = (std::max)(wait_serial, buffer->upload_serial());
    }
    return wait_serial;
}

// The Python-facing present(): everything acquire() promised must still hold,
// and every image this recording samples has to be at least submitted.
inline std::expected<void, Error> present_command_buffer(
    SwapchainRenderer& renderer,
    std::shared_ptr<CommandBuffer> cmd,
    bool capture)
{
    if (auto r = renderer.check_presentable(); !r)
    {
        return std::unexpected(r.error());
    }
    // Claimed here rather than inside record_frame: both submit paths run with
    // the GIL released, so a diagnosis has to travel back as an Error and be
    // raised by the caller, never thrown from under the release.
    if (auto r = cmd->claim_for_frame(renderer.context()->frame_serial()); !r)
    {
        return std::unexpected(r.error());
    }
    auto upload_wait_serial = require_uploads_resident(*cmd);
    if (!upload_wait_serial)
    {
        return std::unexpected(upload_wait_serial.error());
    }
    return renderer.present(std::move(cmd), *upload_wait_serial, capture);
}

// A specialization constant's four bytes, from whichever Python number was
// given. bool is tested FIRST because a Python bool is an int: `True` would
// otherwise arrive as the integer 1, and a shader that declares the constant as
// `bool` reads a different thing from that.
//
// Everything is four bytes, which is what SPIR-V specialization constants of
// scalar type are. A double or an int64 would need a wider block and a second
// map entry size, and no shader here declares one.
inline std::uint32_t spec_constant_bytes(const py::object& value)
{
    std::uint32_t bytes = 0;
    if (py::isinstance<py::bool_>(value))
    {
        // A SPIR-V bool constant is a 32-bit 0 or 1.
        bytes = py::cast<bool>(value) ? 1u : 0u;
    }
    else if (py::isinstance<py::int_>(value))
    {
        const std::int32_t i = py::cast<std::int32_t>(value);
        std::memcpy(&bytes, &i, sizeof(bytes));
    }
    else if (py::isinstance<py::float_>(value))
    {
        const float f = py::cast<float>(value);
        std::memcpy(&bytes, &f, sizeof(bytes));
    }
    else
    {
        raise_error(err_resource(
            "A specialization constant must be a bool, an int or a float. Those are the "
            "scalar types SPIR-V can specialize"));
    }
    return bytes;
}

// The state behind `with cmd.rendering(target, ...):` — carries what
// __enter__/__exit__ need to record the begin/end pair. Deliberately a plain
// struct bound only for its dunder methods; begin_rendering/end_rendering
// stay public, this is sugar, not a replacement.
struct RenderingScope
{
    std::shared_ptr<CommandBuffer> cmd;
    std::shared_ptr<RenderTarget> target;
    std::optional<std::vector<std::array<float, 4>>> clear_color;
    float clear_depth = 1.0f;
    std::uint32_t clear_stencil = 0;
};

// Normalise a Python clear_color into one RGBA per attachment. Accepts both the
// single form [r,g,b,a] (applied to every attachment — the common case) and the
// per-attachment form [[r,g,b,a], …] for MRT. Distinguished by whether the first
// element is itself a sequence.
//
// None is not "clear to black": it is nullopt, which means preserve what the
// attachment holds. An empty vector is still black, so the two stay distinct all
// the way down to the load-op.
inline std::optional<std::vector<std::array<float, 4>>> parse_clear_colors(const py::object& obj)
{
    if (obj.is_none())
    {
        return std::nullopt;
    }
    std::vector<std::array<float, 4>> out;
    py::sequence seq = py::cast<py::sequence>(obj);
    if (py::len(seq) == 0)
    {
        return out;
    }
    auto to_rgba = [](const py::handle& h)
    {
        std::array<float, 4> c{0.0f, 0.0f, 0.0f, 1.0f};
        py::sequence s = py::cast<py::sequence>(h);
        const std::size_t n = py::len(s);
        for (std::size_t i = 0; i < 4 && i < n; ++i)
        {
            c[i] = py::cast<float>(s[i]);
        }
        return c;
    };
    const bool per_attachment = py::isinstance<py::sequence>(seq[0]) && !py::isinstance<py::str>(seq[0]);
    if (per_attachment)
    {
        for (auto item : seq)
        {
            out.push_back(to_rgba(item));
        }
    }
    else
    {
        out.push_back(to_rgba(seq));
    }
    return out;
}

// source= is text or ready SPIR-V, and the Python type is what says which.
// `bytes` has to be tested FIRST: pybind converts both str and bytes to
// std::string, so an `optional<std::string>` parameter would silently compile a
// SPIR-V blob as GLSL and report a syntax error on binary garbage.
//
// The length check is here rather than in the compiler because it is about the
// Python object: a bytes object of the wrong length is not a truncated SPIR-V
// binary, it is the wrong argument.
inline std::optional<ShaderSource> parse_shader_source(const py::object& obj)
{
    if (obj.is_none())
    {
        return std::nullopt;
    }
    if (py::isinstance<py::bytes>(obj))
    {
        const std::string blob = py::cast<std::string>(obj);
        if (blob.size() % sizeof(std::uint32_t) != 0)
        {
            raise_error(err_shader(
                std::format(
                    "source= got {} bytes, which is not a whole number of SPIR-V words. SPIR-V is a stream of "
                    "32-bit words, so its length is always a multiple of 4.",
                    blob.size())));
        }
        std::vector<std::uint32_t> words(blob.size() / sizeof(std::uint32_t));
        std::memcpy(words.data(), blob.data(), blob.size());
        return ShaderSource{std::move(words)};
    }
    return ShaderSource{py::cast<std::string>(obj)};
}

// A GPU timer handle. cmd.timer() records the opening timestamp and returns
// one of these; it is stopped explicitly (stop) or by a `with` (__exit__), and
// read back off itself (ms). The handle IS the identity — no name, no key.
// Holds the command buffer alive so the query pool outlives the handle.
struct Timer
{
    std::shared_ptr<CommandBuffer> cmd;
    std::size_t index = 0;
    std::uint64_t generation = 0;
    bool stopped = false;

    void stop()
    {
        if (!stopped)
        {
            cmd->stop_timer(index); // idempotent: a query written twice would be UB
            stopped = true;
        }
    }
};

// The state behind `with cmd.label("shadow pass"):`. Same shape as
// RenderingScope: a plain struct bound only for its dunders, over verbs that
// stay public.
struct LabelScope
{
    std::shared_ptr<CommandBuffer> cmd;
    std::string name;
};

// An occlusion query handle. Same contract as Timer, deliberately: the handle IS
// the identity, it is closed by stop() or by a `with`, and the result is read
// off itself. Holds the command buffer alive so the query pool outlives it.
struct OcclusionQuery
{
    std::shared_ptr<CommandBuffer> cmd;
    std::size_t index = 0;
    std::uint64_t generation = 0;
    bool stopped = false;

    void stop()
    {
        if (!stopped)
        {
            cmd->stop_occlusion_query(index); // idempotent: ending twice is UB
            stopped = true;
        }
    }
};

// Readback shaped for numpy: (h, w, channels) — or (h, w) for single-channel
// formats, with a leading depth axis for a volume — with the dtype the format
// table dictates. Shared by Image.read and RenderTarget.read_pixels. Mirrors
// the create_image convention: a volume reads back as (d, h, w[, channels]).
inline py::array image_to_numpy(Image& image, std::uint32_t layer = 0, std::uint32_t mip = 0)
{
    auto bytes = unwrap(image.read(/*all_layers=*/false, layer, mip), nullptr);
    const FormatInfo info = format_info(image.format());
    // The shape follows the MIP, not the image: level 2 of a 64x64 texture is
    // 16x16, and returning a 64x64 array with a quarter of it filled would be a
    // shape the caller has to correct for.
    const auto h = static_cast<py::ssize_t>(mip_extent(image.height(), mip));
    const auto w = static_cast<py::ssize_t>(mip_extent(image.width(), mip));
    const auto d = static_cast<py::ssize_t>(mip_extent(image.depth(), mip));

    std::vector<py::ssize_t> shape;
    if (image.is_3d())
    {
        shape = {d, h, w};
    }
    else
    {
        shape = {h, w};
    }
    if (info.channels != 1)
    {
        shape.push_back(static_cast<py::ssize_t>(info.channels));
    }

    py::array out(py::dtype(info.numpy_dtype), shape);
    std::memcpy(out.mutable_data(), bytes.data(), bytes.size());
    return out;
}

// The pixels of one update, validated against the image and packed tightly.
//
// Everything that can be wrong with the caller's array is decided HERE, on the
// main thread with the GIL held, so the worker never has to report a user error
// from a thread that cannot raise. `memcpy` ignores strides, so a non-contiguous
// array is refused with the fix rather than uploaded as garbage — the same rule
// create_image has followed since 0.4.
//
// ResourceError, not ValueError: the image's own format decides each of these, and
// contiguous_nbytes above already answers the identical question about a buffer
// that way. See "Which exception a user error gets" in DESIGN.md.
inline std::vector<std::byte> update_pixels_from_numpy(
    Image& image,
    const py::array& array,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t depth = 1)
{
    const FormatInfo info = format_info(image.format());
    if (info.numpy_dtype[0] == '\0')
    {
        raise_error(err_resource(
            std::format(
                "update() is not available for {}: the format packs several values into one texel, "
                "so no array describes it",
                format_name(image.format()))));
    }
    if (!(array.flags() & py::array::c_style))
    {
        raise_error(err_resource(
            "update(): the array must be C-contiguous (a copy is not made silently, because a "
            "strided array would upload garbage). Use numpy.ascontiguousarray(a)."));
    }
    if (!array.dtype().is(py::dtype(info.numpy_dtype)))
    {
        raise_error(err_resource(
            std::format(
                "update(): this image is {}, so the array must have dtype {}, not {}",
                format_name(image.format()),
                info.numpy_dtype,
                py::str(array.dtype()).cast<std::string>())));
    }

    const py::ssize_t expected_h = static_cast<py::ssize_t>(height);
    const py::ssize_t expected_w = static_cast<py::ssize_t>(width);
    const py::ssize_t expected_d = static_cast<py::ssize_t>(depth);
    const py::ssize_t ndim = array.ndim();
    bool shape_ok = false;
    if (depth > 1)
    {
        // A volume region takes the create_image convention: (d, h, w) for one
        // channel — with a trailing 1 accepted — or (d, h, w, channels).
        const bool dhw = ndim >= 3 && array.shape(0) == expected_d && array.shape(1) == expected_h &&
                         array.shape(2) == expected_w;
        shape_ok = info.channels == 1 ? (dhw && (ndim == 3 || (ndim == 4 && array.shape(3) == 1)))
                                      : (dhw && ndim == 4 && array.shape(3) == static_cast<py::ssize_t>(info.channels));
    }
    else
    {
        shape_ok = info.channels == 1 ? ((ndim == 2 && array.shape(0) == expected_h && array.shape(1) == expected_w) ||
                                         (ndim == 3 && array.shape(0) == expected_h && array.shape(1) == expected_w &&
                                          array.shape(2) == 1))
                                      : (ndim == 3 && array.shape(0) == expected_h && array.shape(1) == expected_w &&
                                         array.shape(2) == static_cast<py::ssize_t>(info.channels));
    }
    if (!shape_ok)
    {
        std::string got;
        for (py::ssize_t i = 0; i < ndim; ++i)
        {
            got += (i ? ", " : "") + std::to_string(array.shape(i));
        }
        std::string expected = depth > 1 ? std::format("({}, {}, {}, {})", depth, height, width, info.channels)
                                         : std::format("({}, {}, {})", height, width, info.channels);
        raise_error(err_resource(
            std::format(
                "update(): expected an array of shape {} for this region of a {} image, got ({})",
                expected,
                format_name(image.format()),
                got)));
    }

    const std::size_t size = static_cast<std::size_t>(width) * height * depth * info.bytes_per_pixel;
    std::vector<std::byte> out(size);
    std::memcpy(out.data(), array.data(), size);
    return out;
}

inline std::expected<void, Error> context_submit(Context& context, std::shared_ptr<CommandBuffer> cmd, bool wait)
{
    // The ring slot this submit is about to record into may still be busy with
    // an earlier asynchronous submit, whose command buffer is the SAME one.
    // Blocking submits have already waited, so this is free for them.
    context.wait_for_slot();

    // Drain hot reloads BEFORE recording, so an edit-then-submit picks up the new
    // pipeline in THIS submit. The headless path advances the ring only after
    // submitting, so hooking the drain there (as the windowed path does) would
    // delay every reload by one submit — and the whole test suite is headless.
    if (auto* hr = context.hot_reload())
    {
        hr->drain();
    }

    // Same claim the windowed present makes. Sequential headless submits can
    // never collide (the ring advances after each one), but mixing a present
    // and a ctx.submit of one CommandBuffer inside a frame can.
    if (auto r = cmd->claim_for_frame(context.frame_serial()); !r)
    {
        return std::unexpected(r.error());
    }
    auto upload_wait_serial = require_uploads_resident(*cmd);
    if (!upload_wait_serial)
    {
        return std::unexpected(upload_wait_serial.error());
    }
    auto recorded = record_frame(*cmd, context);
    if (!recorded)
    {
        return std::unexpected(recorded.error());
    }
    VkCommandBuffer vkCmd = *recorded;

    VkSemaphore timeline = context.submit_timeline();
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    {
        std::lock_guard lock(context.queue_mutex());
        const std::uint64_t serial = context.advance_submit_serial();

        VkTimelineSemaphoreSubmitInfo timelineInfo{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &*upload_wait_serial,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &serial};
        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timelineInfo,
            // A timeline wait for value 0 is trivially satisfied, so this
            // needs no branching on whether uploads are pending.
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &timeline,
            .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1,
            .pCommandBuffers = &vkCmd,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &timeline};

        if (auto e = check(
                context.vk().vkQueueSubmit(context.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE),
                "submit command buffer"))
        {
            return std::unexpected(*e);
        }

        context.note_slot_submit(serial);

        // wait=True is the default and the old behaviour: the next line reads
        // the result, so the round trip is the point. wait=False hands the loop
        // back at once and leaves the pacing to the ring — which is what a
        // compute prototype submitting in a loop wants, because otherwise the
        // GPU idles between iterations and the loop runs at the speed of the
        // round trip rather than of the work.
        // Waits for this submit and nothing else. A vkQueueWaitIdle here would
        // also stall on whatever the upload worker has in flight, which this
        // submit already waited for GPU-side where it mattered.
        if (wait)
        {
            if (auto r = context.wait_for_serial(serial); !r)
            {
                return std::unexpected(r.error());
            }
        }
    }

    // Only meaningful after a wait: the wait above proves everything through
    // this serial is done. An async submit reclaims on the next ctx.wait()
    // instead.
    if (wait)
    {
        context.flush_deletion_queue();
    }

    // A headless submit is a frame too: advance the ring so DynamicBuffer
    // slots and frame descriptor sets rotate (this path used to sit on slot 0
    // forever). After, not before, submitting — an update() made before this
    // call must land in the slot this submit reads.
    context.advance_frame();
    return {};
}

// Attach a debug name to a Vulkan handle (empty name -> no-op). Non-dispatchable
// handles are 64-bit; reinterpret_cast is the Vulkan-idiomatic conversion on the
// 64-bit builds bazalt ships.
template <typename Handle>
void name_object(Context& ctx, VkObjectType type, Handle handle, const std::string& name)
{
    if (name.empty())
    {
        return;
    }
    ctx.set_debug_name(type, reinterpret_cast<std::uint64_t>(handle), name);
}

// A DynamicBuffer is one VkBuffer per in-flight frame; name each the same (a
// StaticBuffer hands out the same handle for every frame, harmlessly re-named).
inline void name_buffer(Context& ctx, const std::shared_ptr<Buffer>& buffer, const std::string& name)
{
    if (name.empty())
    {
        return;
    }
    for (std::uint32_t i = 0; i < ctx.frames_in_flight(); ++i)
    {
        name_object(ctx, VK_OBJECT_TYPE_BUFFER, buffer->get_for_frame(i), name);
    }
}
