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

#ifdef _WIN32
#include <windows.h>
#endif

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

// Renders a command buffer into whatever targets it captured, with no swapchain
// and no present. This is the headless path — and the one the test suite uses.
std::expected<void, Error> context_submit(Context& context, std::shared_ptr<CommandBuffer> cmd, bool wait);

// ── Error boundary ────────────────────────────────────────────────────────────
//
// Every ErrorCode gets its own Python class so that recoverability is expressible
// as `except bz.ShaderError`. Previously every failure — a shader typo and a lost
// device alike — arrived as a bare RuntimeError, so callers had no way to keep
// running after the recoverable ones.

namespace
{

    py::handle exc_bazalt;
    py::handle exc_initialization;
    py::handle exc_device_lost;
    py::handle exc_out_of_memory;
    py::handle exc_shader;
    py::handle exc_window;
    py::handle exc_resource;

    py::handle make_exception(py::module_& m, const char* name, py::handle base)
    {
        std::string qualified = std::string("bazalt._core.") + name;
        py::object exc = py::reinterpret_steal<py::object>(PyErr_NewException(qualified.c_str(), base.ptr(), nullptr));
        m.add_object(name, exc);
        return exc.release();
    }

    void register_exceptions(py::module_& m)
    {
        exc_bazalt = make_exception(m, "BazaltError", PyExc_Exception);
        exc_initialization = make_exception(m, "InitializationError", exc_bazalt);
        exc_device_lost = make_exception(m, "DeviceLostError", exc_bazalt);
        exc_out_of_memory = make_exception(m, "OutOfMemoryError", exc_bazalt);
        exc_shader = make_exception(m, "ShaderError", exc_bazalt);
        exc_window = make_exception(m, "WindowError", exc_bazalt);
        exc_resource = make_exception(m, "ResourceError", exc_bazalt);
    }

    py::handle exception_for(ErrorCode code)
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
        }
        return exc_bazalt;
    }

    [[noreturn]] void raise_error(const Error& error)
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

    void unwrap(std::expected<void, Error>&& result, Logger* logger)
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
    void require_same_context(const Context* a, const Context* b, const char* what)
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
    void require_preservable(const RenderTarget& target, bool preserve, const char* what)
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

    // Resolves a list's element type from the explicit argument or the first
    // element. `int_default` is the caller's policy: create_buffer infers UINT32
    // for integers going into an INDEX buffer, update infers INT32 — a deliberate
    // difference, not drift.
    DataType resolve_data_type(const py::list& list, std::optional<DataType> requested, DataType int_default)
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
    bool is_c_contiguous(const py::buffer_info& info)
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
    size_t contiguous_nbytes(const py::buffer_info& info, const char* what)
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

    // One (h, w) or (h, w, channels) array → its GPU Format + dimensions. UNORM,
    // never sRGB (arrays are data, files are pictures). Raises a ResourceError on
    // an unsupported dtype/shape. Shared by the single-image and layered
    // (texture array / cubemap) create_image paths.
    struct ArrayImageSpec
    {
        Format format;
        uint32_t width;
        uint32_t height;
    };
    ArrayImageSpec array_image_spec(const py::buffer_info& info)
    {
        if (info.ndim != 2 && info.ndim != 3)
        {
            raise_error(err_resource(
                std::format("create_image expects a (h, w) or (h, w, channels) array, got {} dimensions", info.ndim)));
        }
        const auto height = static_cast<uint32_t>(info.shape[0]);
        const auto width = static_cast<uint32_t>(info.shape[1]);
        const py::ssize_t channels = info.ndim == 3 ? info.shape[2] : 1;

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
            raise_error(err_resource(
                std::format(
                    "create_image: unsupported dtype/shape (dtype '{}', {} channel(s)). "
                    "Supported: uint8 x 1/2/4 channels, float16 x 1/4, float32 x 1/4",
                    info.format,
                    channels)));
        }
        return {*format, width, height};
    }

    const char* severity_name(Severity severity)
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

    ValidationMode parse_validation(const std::string& value)
    {
        if (value == "auto")
            return ValidationMode::Auto;
        if (value == "on")
            return ValidationMode::On;
        if (value == "off")
            return ValidationMode::Off;
        if (value == "sync")
            return ValidationMode::Sync;
        throw std::invalid_argument(
            std::format("validation must be one of 'auto', 'on', 'off', 'sync' (got '{}')", value));
    }

    // A Context built without a logger used to render with validation off and say
    // nothing about its own failures. Default to reporting warnings on stderr.
    std::shared_ptr<Logger> make_default_logger()
    {
        auto logger = std::make_shared<Logger>(Severity::Warning);
        logger->register_callback(
            py::cpp_function(
                [](const LogMessage& msg)
                {
                    py::object stderr_stream = py::module_::import("sys").attr("stderr");
                    stderr_stream.attr("write")(
                        std::format("[bazalt] {}: {}\n", severity_name(msg.severity), msg.text));
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
    std::expected<VkCommandBuffer, Error> record_frame(
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

} // namespace
std::expected<void, Error> SwapchainRenderer::present(
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
std::expected<std::uint64_t, Error> require_uploads_resident(CommandBuffer& cmd)
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
std::expected<void, Error> present_command_buffer(
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
static std::uint32_t spec_constant_bytes(const py::object& value)
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
static std::optional<std::vector<std::array<float, 4>>> parse_clear_colors(const py::object& obj)
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
static std::optional<ShaderSource> parse_shader_source(const py::object& obj)
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
// formats — with the dtype the format table dictates. Shared by Image.read and
// RenderTarget.read_pixels.
py::array image_to_numpy(Image& image, std::uint32_t layer = 0, std::uint32_t mip = 0)
{
    auto bytes = unwrap(image.read(/*all_layers=*/false, layer, mip), nullptr);
    const FormatInfo info = format_info(image.format());
    // The shape follows the MIP, not the image: level 2 of a 64x64 texture is
    // 16x16, and returning a 64x64 array with a quarter of it filled would be a
    // shape the caller has to correct for.
    const auto h = static_cast<py::ssize_t>(mip_extent(image.height(), mip));
    const auto w = static_cast<py::ssize_t>(mip_extent(image.width(), mip));

    std::vector<py::ssize_t> shape;
    if (info.channels == 1)
    {
        shape = {h, w};
    }
    else
    {
        shape = {h, w, static_cast<py::ssize_t>(info.channels)};
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
std::vector<std::byte> update_pixels_from_numpy(
    Image& image,
    const py::array& array,
    std::uint32_t width,
    std::uint32_t height)
{
    const FormatInfo info = format_info(image.format());
    if (info.numpy_dtype[0] == '\0')
    {
        throw py::value_error(
            std::format(
                "update() is not available for {}: the format packs several values into one texel, "
                "so no array describes it",
                format_name(image.format())));
    }
    if (!(array.flags() & py::array::c_style))
    {
        throw py::value_error(
            "update(): the array must be C-contiguous (a copy is not made silently, because a "
            "strided array would upload garbage). Use numpy.ascontiguousarray(a).");
    }
    if (!array.dtype().is(py::dtype(info.numpy_dtype)))
    {
        throw py::value_error(
            std::format(
                "update(): this image is {}, so the array must have dtype {}, not {}",
                format_name(image.format()),
                info.numpy_dtype,
                py::str(array.dtype()).cast<std::string>()));
    }

    const py::ssize_t expected_h = static_cast<py::ssize_t>(height);
    const py::ssize_t expected_w = static_cast<py::ssize_t>(width);
    const py::ssize_t ndim = array.ndim();
    const bool shape_ok =
        info.channels == 1
            ? ((ndim == 2 && array.shape(0) == expected_h && array.shape(1) == expected_w) ||
               (ndim == 3 && array.shape(0) == expected_h && array.shape(1) == expected_w && array.shape(2) == 1))
            : (ndim == 3 && array.shape(0) == expected_h && array.shape(1) == expected_w &&
               array.shape(2) == static_cast<py::ssize_t>(info.channels));
    if (!shape_ok)
    {
        std::string got;
        for (py::ssize_t i = 0; i < ndim; ++i)
        {
            got += (i ? ", " : "") + std::to_string(array.shape(i));
        }
        throw py::value_error(
            std::format(
                "update(): expected an array of shape ({}, {}, {}) for a {} region of a {} image, got ({})",
                height,
                width,
                info.channels,
                format_name(image.format()),
                format_name(image.format()),
                got));
    }

    const std::size_t size = static_cast<std::size_t>(width) * height * info.bytes_per_pixel;
    std::vector<std::byte> out(size);
    std::memcpy(out.data(), array.data(), size);
    return out;
}

std::expected<void, Error> context_submit(Context& context, std::shared_ptr<CommandBuffer> cmd, bool wait)
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
void name_buffer(Context& ctx, const std::shared_ptr<Buffer>& buffer, const std::string& name)
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

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Bazalt native core module";

    // Logger drain threads call into Python; joining them after the
    // interpreter starts finalizing crashes ("could not acquire lock for
    // stderr at interpreter shutdown"). atexit runs while Python is intact,
    // so every script that ends with a live Context exits cleanly.
    py::module_::import("atexit").attr("register")(py::cpp_function([]() { Logger::shutdown_all(); }));

    register_exceptions(m);

    // py::arithmetic() so `msg.severity >= bz.Severity.WARNING` works — filtering
    // by level is the whole point of carrying one.
    py::enum_<Severity>(m, "Severity", py::arithmetic())
        .value("INFO", Severity::Info)
        .value("WARNING", Severity::Warning)
        .value("ERROR", Severity::Error);

    py::enum_<Source>(m, "Source")
        .value("GENERAL", Source::General)
        .value("VALIDATION", Source::Validation)
        .value("WINDOW", Source::Window)
        .value("SHADER", Source::Shader)
        .value("UPLOAD", Source::Upload)
        .value("DEVICE", Source::Device);

    py::class_<LogMessage>(m, "LogMessage")
        .def_readonly("severity", &LogMessage::severity)
        .def_readonly("source", &LogMessage::source)
        .def_readonly("text", &LogMessage::text)
        .def(
            "__str__",
            [](const LogMessage& msg) { return std::format("{}: {}", severity_name(msg.severity), msg.text); })
        .def(
            "__repr__",
            [](const LogMessage& msg)
            { return std::format("<LogMessage {} '{}'>", severity_name(msg.severity), msg.text); });

    // Capabilities, not versions/extensions: the same capability has different
    // spellings per driver (dynamic rendering is an extension on 1.2, core in
    // 1.3), so which one to use is bazalt's problem, not the user's. New entries
    // here are additive, so nothing about this needs to wait for a 2.0.
    py::enum_<Feature>(m, "Feature")
        .value("ANISOTROPIC_FILTERING", Feature::ANISOTROPIC_FILTERING)
        .value("WIREFRAME", Feature::WIREFRAME)
        .value("WIDE_LINES", Feature::WIDE_LINES)
        .value("DEPTH_CLAMP", Feature::DEPTH_CLAMP)
        .value("SAMPLE_RATE_SHADING", Feature::SAMPLE_RATE_SHADING)
        .value("MULTI_DRAW_INDIRECT", Feature::MULTI_DRAW_INDIRECT)
        .value("SHADER_FLOAT64", Feature::SHADER_FLOAT64)
        .value("INDEPENDENT_BLEND", Feature::INDEPENDENT_BLEND)
        .value("TESSELLATION", Feature::TESSELLATION)
        .value("GEOMETRY_SHADER", Feature::GEOMETRY_SHADER)
        .value("FRAGMENT_STORES", Feature::FRAGMENT_STORES)
        .value("VERTEX_STAGE_STORES", Feature::VERTEX_STAGE_STORES);

    py::enum_<BufferType>(m, "BufferType")
        .value("VERTEX", BufferType::VERTEX)
        .value("INDEX", BufferType::INDEX)
        .value("UNIFORM", BufferType::UNIFORM)
        .value("STORAGE", BufferType::STORAGE);

    py::enum_<DataType>(m, "DataType")
        .value("FLOAT", DataType::FLOAT)
        .value("UINT32", DataType::UINT32)
        .value("UINT16", DataType::UINT16)
        .value("INT32", DataType::INT32);

    py::enum_<ShaderStage>(m, "ShaderStage")
        .value("VERTEX", ShaderStage::VERTEX)
        .value("FRAGMENT", ShaderStage::FRAGMENT)
        .value("COMPUTE", ShaderStage::COMPUTE)
        .value("TESS_CONTROL", ShaderStage::TESS_CONTROL)
        .value("TESS_EVALUATION", ShaderStage::TESS_EVALUATION)
        .value("GEOMETRY", ShaderStage::GEOMETRY);

    py::enum_<VertexFormat>(m, "VertexFormat")
        .value("FLOAT2", VertexFormat::FLOAT2)
        .value("FLOAT3", VertexFormat::FLOAT3)
        .value("FLOAT4", VertexFormat::FLOAT4)
        .value("FLOAT", VertexFormat::FLOAT)
        .value("UBYTE4_NORM", VertexFormat::UBYTE4_NORM)
        .value("UINT", VertexFormat::UINT);

    py::enum_<Topology>(m, "Topology")
        .value("TRIANGLE_LIST", Topology::TRIANGLE_LIST)
        .value("POINT_LIST", Topology::POINT_LIST)
        .value("LINE_LIST", Topology::LINE_LIST)
        .value("TRIANGLE_STRIP", Topology::TRIANGLE_STRIP)
        .value("LINE_STRIP", Topology::LINE_STRIP)
        .value("PATCH_LIST", Topology::PATCH_LIST);

    // The vocabulary of cmd.barrier() in manual mode (auto_barriers=False).
    py::enum_<Access>(m, "Access")
        .value("SHADER_READ", Access::SHADER_READ)
        .value("SHADER_WRITE", Access::SHADER_WRITE)
        .value("VERTEX_READ", Access::VERTEX_READ)
        .value("INDEX_READ", Access::INDEX_READ)
        .value("UNIFORM_READ", Access::UNIFORM_READ)
        .value("INDIRECT_READ", Access::INDIRECT_READ);

    // Pixel formats — the name VertexFormat freed in 0.4.
    py::enum_<Format>(m, "Format")
        .value("RGBA8", Format::RGBA8)
        .value("RGBA8_SRGB", Format::RGBA8_SRGB)
        .value("BGRA8", Format::BGRA8)
        .value("R8", Format::R8)
        .value("RG8", Format::RG8)
        .value("R16F", Format::R16F)
        .value("RGBA16F", Format::RGBA16F)
        .value("R32F", Format::R32F)
        .value("RGBA32F", Format::RGBA32F)
        .value("D32F", Format::D32F)
        .value("R32_UINT", Format::R32_UINT)
        .value("R11G11B10F", Format::R11G11B10F)
        .value("DEPTH_STENCIL", Format::DEPTH_STENCIL);

    py::enum_<Filter>(m, "Filter").value("LINEAR", Filter::LINEAR).value("NEAREST", Filter::NEAREST);

    py::enum_<AddressMode>(m, "AddressMode")
        .value("REPEAT", AddressMode::REPEAT)
        .value("CLAMP", AddressMode::CLAMP)
        .value("MIRROR", AddressMode::MIRROR)
        .value("CLAMP_TO_BORDER", AddressMode::CLAMP_TO_BORDER);

    py::enum_<BorderColor>(m, "BorderColor")
        .value("OPAQUE_BLACK", BorderColor::OPAQUE_BLACK)
        .value("OPAQUE_WHITE", BorderColor::OPAQUE_WHITE);

    py::enum_<CompareOp>(m, "CompareOp")
        .value("NEVER", CompareOp::NEVER)
        .value("LESS", CompareOp::LESS)
        .value("EQUAL", CompareOp::EQUAL)
        .value("LESS_OR_EQUAL", CompareOp::LESS_OR_EQUAL)
        .value("GREATER", CompareOp::GREATER)
        .value("NOT_EQUAL", CompareOp::NOT_EQUAL)
        .value("GREATER_OR_EQUAL", CompareOp::GREATER_OR_EQUAL)
        .value("ALWAYS", CompareOp::ALWAYS);

    py::enum_<StencilOp>(m, "StencilOp")
        .value("KEEP", StencilOp::KEEP)
        .value("ZERO", StencilOp::ZERO)
        .value("REPLACE", StencilOp::REPLACE)
        .value("INCREMENT_CLAMP", StencilOp::INCREMENT_CLAMP)
        .value("DECREMENT_CLAMP", StencilOp::DECREMENT_CLAMP)
        .value("INVERT", StencilOp::INVERT)
        .value("INCREMENT_WRAP", StencilOp::INCREMENT_WRAP)
        .value("DECREMENT_WRAP", StencilOp::DECREMENT_WRAP);

    py::enum_<BlendMode>(m, "BlendMode")
        .value("ALPHA", BlendMode::ALPHA)
        .value("ADDITIVE", BlendMode::ADDITIVE)
        .value("PREMULTIPLIED", BlendMode::PREMULTIPLIED);

    py::enum_<PolygonMode>(m, "PolygonMode")
        .value("FILL", PolygonMode::FILL)
        .value("LINE", PolygonMode::LINE)
        .value("POINT", PolygonMode::POINT);

    py::enum_<CullMode>(m, "CullMode")
        .value("NONE", CullMode::NONE)
        .value("BACK", CullMode::BACK)
        .value("FRONT", CullMode::FRONT)
        .value("FRONT_AND_BACK", CullMode::FRONT_AND_BACK);

    py::enum_<FrontFace>(m, "FrontFace")
        .value("CLOCKWISE", FrontFace::CLOCKWISE)
        .value("COUNTER_CLOCKWISE", FrontFace::COUNTER_CLOCKWISE);

    py::enum_<MemoryUsage>(m, "MemoryUsage")
        .value("STATIC", MemoryUsage::STATIC)
        .value("DYNAMIC", MemoryUsage::DYNAMIC);

    py::class_<MouseState>(m, "MouseState")
        .def_readonly("x", &MouseState::x)
        .def_readonly("y", &MouseState::y)
        .def_readonly("dx", &MouseState::dx)
        .def_readonly("dy", &MouseState::dy)
        .def_readonly("scroll_dx", &MouseState::scroll_dx)
        .def_readonly("scroll_dy", &MouseState::scroll_dy);

    py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer")
        // offset= is a byte offset into the buffer. Without it, changing one
        // matrix of an instance array rewrites the whole array — the update is
        // already the per-frame path, so the wasted copy is per frame too.
        .def(
            "update",
            [](Buffer& buffer, std::string_view data, size_t offset)
            { unwrap(buffer.update(std::as_bytes(std::span(data.data(), data.size())), offset), nullptr); },
            py::arg("data"),
            py::kw_only(),
            py::arg("offset") = 0)
        .def(
            "update",
            [](Buffer& buffer, py::buffer b, size_t offset)
            {
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
                if (list.empty())
                    return;
                DataType actualType = resolve_data_type(list, dataType, DataType::INT32);
                with_list_bytes(
                    list,
                    actualType,
                    [&](const void* data, size_t nbytes)
                    { unwrap(buffer.update({static_cast<const std::byte*>(data), nbytes}, offset), nullptr); });
            },
            py::arg("list"),
            py::arg("data_type") = py::none(),
            py::kw_only(),
            py::arg("offset") = 0)
        // dtype is mandatory: buffers carry no format (unlike Images), so the
        // caller has to say how to interpret the bytes.
        .def(
            "read",
            [](Buffer& self, py::object dtype) -> py::array
            {
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
            { return image_to_numpy(self, layer, mip); },
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
                Context* context = const_cast<Context*>(self->owner());
                if (!context)
                {
                    throw py::value_error("update(): this image has no Context");
                }
                if (self->samples() != 1)
                {
                    throw py::value_error(
                        "update(): a multisampled image cannot be uploaded to. It is rendered "
                        "into and resolved out");
                }
                if (layer >= self->array_layers())
                {
                    throw py::value_error(
                        std::format("update(layer={}): this image has {} layer(s)", layer, self->array_layers()));
                }
                if (mip >= self->mip_levels())
                {
                    throw py::value_error(
                        std::format("update(mip={}): this image has {} mip level(s)", mip, self->mip_levels()));
                }

                const std::uint32_t level_w = mip_extent(self->width(), mip);
                const std::uint32_t level_h = mip_extent(self->height(), mip);
                std::uint32_t x = 0, y = 0, w = level_w, h = level_h;
                if (!region.is_none())
                {
                    py::sequence seq = py::cast<py::sequence>(region);
                    if (py::len(seq) != 4)
                    {
                        throw py::value_error("update(region=): expected (x, y, width, height)");
                    }
                    x = py::cast<std::uint32_t>(seq[0]);
                    y = py::cast<std::uint32_t>(seq[1]);
                    w = py::cast<std::uint32_t>(seq[2]);
                    h = py::cast<std::uint32_t>(seq[3]);
                    if (w == 0 || h == 0 || !fits_within(x, w, level_w) || !fits_within(y, h, level_h))
                    {
                        throw py::value_error(
                            std::format(
                                "update(region=({}, {}, {}, {})): does not fit in the {}x{} of mip {}",
                                x,
                                y,
                                w,
                                h,
                                level_w,
                                level_h,
                                mip));
                    }
                }

                std::vector<std::byte> pixels = update_pixels_from_numpy(*self, array, w, h);

                auto* manager = static_cast<UploadManager*>(context->upload_manager());
                manager->update(
                    std::move(self),
                    std::move(pixels),
                    layer,
                    mip,
                    VkOffset2D{static_cast<std::int32_t>(x), static_cast<std::int32_t>(y)},
                    VkExtent2D{w, h});
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
            [](GraphicsPipelineBuilder& self, uint32_t binding, ShaderStage stage, uint32_t set)
                -> GraphicsPipelineBuilder& { return self.uniform_buffer(binding, stage, set); },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set"))
        .def(
            "storage_buffer",
            [](GraphicsPipelineBuilder& self, uint32_t binding, ShaderStage stage, uint32_t set)
                -> GraphicsPipelineBuilder& { return self.storage_buffer(binding, stage, set); },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set"))
        .def(
            "texture",
            [](GraphicsPipelineBuilder& self, uint32_t binding, ShaderStage stage, uint32_t set)
                -> GraphicsPipelineBuilder& { return self.texture(binding, stage, set); },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set"))
        .def(
            "storage_image",
            [](GraphicsPipelineBuilder& self, uint32_t binding, ShaderStage stage, uint32_t set)
                -> GraphicsPipelineBuilder& { return self.storage_image(binding, stage, set); },
            py::arg("binding"),
            py::arg("stage"),
            py::arg("set"))
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
            [](ComputePipelineBuilder& self, uint32_t binding, uint32_t set) -> ComputePipelineBuilder&
            { return self.uniform_buffer(binding, set); },
            py::arg("binding"),
            py::arg("set") = 0)
        .def(
            "storage_buffer",
            [](ComputePipelineBuilder& self, uint32_t binding, uint32_t set) -> ComputePipelineBuilder&
            { return self.storage_buffer(binding, set); },
            py::arg("binding"),
            py::arg("set") = 0)
        .def(
            "storage_image",
            [](ComputePipelineBuilder& self, uint32_t binding, uint32_t set) -> ComputePipelineBuilder&
            { return self.storage_image(binding, set); },
            py::arg("binding"),
            py::arg("set") = 0)
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
        .def(
            "set_image",
            [](DescriptorSet& self, uint32_t binding, std::shared_ptr<Image> image, std::shared_ptr<Sampler> sampler)
            {
                require_same_context(self.owner(), image->owner(), "set_image");
                unwrap(self.set_image(binding, std::move(image), std::move(sampler)), nullptr);
            },
            py::arg("binding"),
            py::arg("image"),
            py::arg("sampler") = py::none())
        .def(
            "set_storage_image",
            [](DescriptorSet& self, uint32_t binding, std::shared_ptr<Image> image)
            {
                require_same_context(self.owner(), image->owner(), "set_storage_image");
                unwrap(self.set_storage_image(binding, std::move(image)), nullptr);
            },
            py::arg("binding"),
            py::arg("image"))
        .def(
            "set_buffer",
            [](DescriptorSet& self, uint32_t binding, std::shared_ptr<Buffer> buffer)
            {
                require_same_context(self.owner(), buffer->owner(), "set_buffer");
                unwrap(self.set_buffer(binding, std::move(buffer)), nullptr);
            },
            py::arg("binding"),
            py::arg("buffer"));

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
               std::uint32_t count)
            {
                require_same_context(self->owner(), buffer->owner(), "draw_indirect");
                unwrap(self->draw_indirect(std::move(buffer), offset, count), nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("offset") = 0,
            py::arg("count") = 1)
        .def(
            "draw_indexed_indirect",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Buffer> buffer,
               VkDeviceSize offset,
               std::uint32_t count)
            {
                require_same_context(self->owner(), buffer->owner(), "draw_indexed_indirect");
                unwrap(self->draw_indexed_indirect(std::move(buffer), offset, count), nullptr);
                return self;
            },
            py::arg("buffer"),
            py::arg("offset") = 0,
            py::arg("count") = 1)
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
        .def_property_readonly(
            "samples",
            [](const OcclusionQuery& self) { return self.cmd->read_occlusion_query(self.index, self.generation); });

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
        .def_property_readonly(
            "ms", [](const Timer& self) { return self.cmd->read_timer(self.index, self.generation); });

    // ── Window (GLFW) ──
    py::enum_<WindowMode>(m, "WindowMode")
        .value("WINDOWED", WindowMode::WINDOWED)
        .value("FRAMELESS", WindowMode::FRAMELESS)
        .value("FULLSCREEN", WindowMode::FULLSCREEN)
        .value("FULLSCREEN_WINDOWED", WindowMode::FULLSCREEN_WINDOWED);

    py::class_<Window>(m, "Window")
        .def(
            py::init(
                [](int width, int height, const std::string& title, std::shared_ptr<Logger> logger, WindowMode mode)
                {
                    // Window used to have no way to reach a Logger at all, so GLFW's own
                    // diagnostics went nowhere.
                    return unwrap(Window::create(width, height, title, logger, mode), logger.get());
                }),
            py::arg("width"),
            py::arg("height"),
            py::arg("title"),
            py::arg("logger") = py::none(),
            py::arg("mode") = WindowMode::WINDOWED)
        .def("is_open", &Window::is_open)
        .def("is_key_pressed", &Window::is_key_pressed, py::arg("key"))
        .def("is_mouse_button_pressed", &Window::is_mouse_button_pressed, py::arg("button"))
        .def("was_key_pressed", &Window::was_key_pressed, py::arg("key"))
        .def("was_mouse_button_pressed", &Window::was_mouse_button_pressed, py::arg("button"))
        .def("set_cursor_mode", &Window::set_cursor_mode, py::arg("mode"))
        .def("get_mouse_state", &Window::get_mouse_state)
        .def("set_title", &Window::set_title, py::arg("title"))
        .def(
            "set_mode",
            // nullptr logger: GLFW's own error callback has already logged the
            // reason through the Window's Logger, so passing it here would say
            // the same thing twice.
            [](Window& self, WindowMode mode) { unwrap(self.set_mode(mode), nullptr); },
            py::arg("mode"))
        .def("set_size", &Window::set_size, py::arg("width"), py::arg("height"))
        .def("set_position", &Window::set_position, py::arg("x"), py::arg("y"))
        .def("set_cursor_position", &Window::set_cursor_position, py::arg("x"), py::arg("y"))
        .def(
            "dropped_files",
            // Copied into a Python list rather than returned by reference: the
            // vector is rotated out from under the caller on the next poll cycle.
            [](const Window& self) { return py::cast(self.dropped_files()); })
        .def(
            "set_icon",
            [](Window& self, py::object icon)
            {
                if (icon.is_none())
                {
                    self.set_icon({}, 0, 0);
                    return;
                }
                // Validated here, in the binding, for the same reason create_image
                // validates here: this is a user error about the shape of a Python
                // object, the GIL is held, and raise_error is legal.
                auto array = icon.cast<py::array>();
                // Compared against the dtype object, like image.update does, rather
                // than against a kind character: numpy spells uint8's kind 'u' and
                // its char code 'B', and a hand-written check picks the wrong one.
                if (!array.dtype().is(py::dtype("uint8")))
                {
                    raise_error(err_window(
                        std::format(
                            "set_icon needs an RGBA8 array of dtype uint8, not {}. Convert it with "
                            "arr.astype(np.uint8)",
                            py::str(array.dtype()).cast<std::string>())));
                }
                // Two conditions, two messages: "wrong number of dimensions" and
                // "no alpha channel" are different mistakes, and one message
                // covering both names neither.
                if (array.ndim() != 3)
                {
                    raise_error(err_window(
                        std::format(
                            "set_icon needs a (height, width, 4) RGBA array, got {} dimensions", array.ndim())));
                }
                if (array.shape(2) != 4)
                {
                    raise_error(err_window(
                        std::format(
                            "set_icon needs 4 channels (RGBA), got {}. An icon has an alpha channel", array.shape(2))));
                }
                // memcpy ignores strides, so a view like arr[::2] or arr.T would
                // copy other bytes. The 0.4 rule, applied again.
                if (!(array.flags() & py::array::c_style))
                {
                    raise_error(err_window(
                        "set_icon: the array must be C-contiguous (a strided view like arr.T "
                        "or arr[::2] would copy other bytes). Use numpy.ascontiguousarray(a)."));
                }
                const auto height = static_cast<int>(array.shape(0));
                const auto width = static_cast<int>(array.shape(1));
                const auto* bytes = static_cast<const std::uint8_t*>(array.data());
                self.set_icon(std::vector<std::uint8_t>(bytes, bytes + array.nbytes()), width, height);
            },
            py::arg("icon"))
        .def("set_resizable", &Window::set_resizable, py::arg("enable"))
        .def("set_always_on_top", &Window::set_always_on_top, py::arg("enable"))
        .def("set_opacity", &Window::set_opacity, py::arg("opacity"))
        .def_property_readonly("mode", &Window::mode)
        .def_property_readonly("position", &Window::get_position)
        .def_property_readonly("resizable", &Window::is_resizable)
        .def_property_readonly("always_on_top", &Window::is_always_on_top)
        .def_property_readonly("opacity", &Window::get_opacity)
        .def_property_readonly("content_scale", &Window::get_content_scale)
        .def_property_readonly("width", &Window::get_width)
        .def_property_readonly("height", &Window::get_height);

    // ── Logger ──
    py::class_<Logger, std::shared_ptr<Logger>>(m, "Logger")
        .def(py::init<Severity>(), py::arg("min_severity") = Severity::Warning)
        // One callback receiving a structured LogMessage, not on_error/on_warning/
        // on_info. Three callbacks would be three ways to do one thing, and the old
        // on_error was a lie anyway — it received INFO and WARNING alike.
        .def(
            "on_message",
            [](Logger& self, py::function callback)
            {
                self.register_callback(callback);
                return callback; // returned so it works as a decorator
            },
            py::arg("callback"))
        .def(
            "log",
            [](Logger& self, const std::string& text, Severity severity, Source source)
            { self.log(severity, source, text); },
            py::arg("text"),
            py::arg("severity") = Severity::Info,
            py::arg("source") = Source::General)
        // Delivery is async; without flush(), asserting "no errors happened" only
        // asserts "none had arrived yet".
        .def("flush", &Logger::flush)
        .def_property("min_severity", &Logger::min_severity, &Logger::set_min_severity);

    // ── Context ──
    // ── Device ──
    // Inert data, not a live handle: see Device.hpp on why a VkPhysicalDevice
    // could not survive the enumeration that produced it.
    py::class_<Device>(m, "Device")
        .def_readonly("name", &Device::name)
        .def_readonly("type", &Device::type)
        .def_property_readonly("api_version", [](const Device& d) { return api_version_string(d.api_version); })
        // Megabytes rather than bytes: the number is read by a human choosing a
        // card, and "8188" beats "8584495104".
        .def_property_readonly(
            "memory_mb", [](const Device& d) { return static_cast<std::uint64_t>(d.memory_bytes / (1024 * 1024)); })
        .def("supports", &Device::supports, py::arg("feature"))
        .def("supports_multiview", [](const Device& d) { return d.multiview; })
        .def(
            "__repr__",
            [](const Device& d)
            {
                return std::format("<bazalt.Device '{}' ({}, {} MB)>", d.name, d.type, d.memory_bytes / (1024 * 1024));
            });

    // Free function, not a Window method: GLFW's event queue is process-wide.
    m.def(
        "poll_events",
        []() { unwrap(poll_events(), nullptr); },
        "Drain the OS event queue and dispatch each event to the window it was\n"
        "addressed to. One call services every window. The per-window distinction\n"
        "lives in the queries (is_key_pressed, is_open, renderer.acquire).\n"
        "Raises WindowError when no window exists.");

    // Free functions for the same reason poll_events is one: the clipboard belongs
    // to the process and the GLFW calls take no window.
    m.def(
        "get_clipboard",
        []() { return unwrap(get_clipboard(), nullptr); },
        "The system clipboard as text, or an empty string when it holds nothing or\n"
        "holds something that is not text. Needs at least one live Window, because\n"
        "GLFW is initialized with the first one.");

    m.def(
        "set_clipboard",
        [](const std::string& text) { unwrap(set_clipboard(text), nullptr); },
        py::arg("text"),
        "Put text on the system clipboard. Needs at least one live Window.");

    m.def(
        "list_devices",
        []()
        {
            auto devices = list_devices();
            return unwrap(std::move(devices), nullptr);
        },
        "Every GPU on this machine, without creating a Context. Pass one to\n"
        "Context(device=...) to run on it. The default picks automatically.");

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
                    // An argument-validity error, so ValueError — matching what
                    // validation="nonsense" raises, not the BazaltError hierarchy.
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
        .def("supports_multiview", &Context::supports_multiview)
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
            py::arg("list"),
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
            py::arg("array"),
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
            py::arg("size_in_bytes"),
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
               uint32_t layers,
               bool cube,
               uint32_t mip_levels,
               const std::string& name) -> py::object
            {
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
                if (width > 0 && height > 0)
                {
                    const uint32_t max_mips = Image::full_mip_count(width, height);
                    if (mip_levels < 1 || mip_levels > max_mips)
                    {
                        raise_error(err_resource(
                            std::format(
                                "create_image: mip_levels must be 1..{} for a {}x{} image, got {}",
                                max_mips,
                                width,
                                height,
                                mip_levels)));
                    }
                }
                auto image = unwrap(
                    Image::create_empty(self, width, height, format, mip_levels, layers, cube), self.logger().get());
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                return py::cast(image);
            },
            py::arg("width"),
            py::arg("height"),
            py::arg("format") = Format::RGBA8,
            py::kw_only(),
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
                const ArrayImageSpec spec = array_image_spec(info);
                auto image = unwrap(
                    Image::create_from_pixels(self, info.ptr, spec.width, spec.height, spec.format, mipmaps),
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

                auto image =
                    layers > 1
                        ? unwrap(
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
                                  self, pixels.data(), source->width(), source->height(), source->format(), mipmaps),
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
                                VkOffset2D{0, 0},
                                VkExtent2D{w, h});
                        }
                    }
                }
                name_object(self, VK_OBJECT_TYPE_IMAGE, image->vk_image(), name);
                return py::cast(image);
            },
            py::arg("source"),
            py::kw_only(),
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
               uint32_t maxSets,
               uint32_t samplers,
               uint32_t uniformBuffers,
               uint32_t storageBuffers,
               uint32_t storageImages) -> py::object
            {
                return py::cast(unwrap(
                    DescriptorPool::create(self, maxSets, samplers, uniformBuffers, storageBuffers, storageImages),
                    self.logger().get()));
            },
            py::arg("max_sets"),
            py::arg("samplers") = 0,
            py::arg("uniform_buffers") = 0,
            py::arg("storage_buffers") = 0,
            py::arg("storage_images") = 0)
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

    // ── RenderTarget ──
    py::class_<RenderTarget, std::shared_ptr<RenderTarget>>(m, "RenderTargetBase");

    py::class_<OffscreenTarget, RenderTarget, std::shared_ptr<OffscreenTarget>>(m, "RenderTarget")
        // color: None | Format | [Format, ...]; depth: None | Format. A bool
        // depth is refused with a migration hint (it was one in 0.4).
        .def(
            py::init(
                [](Context& context,
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
                                // overload below but also passed width/height, which that
                                // overload does not take. pybind cannot fall through to it
                                // once this signature has matched, so say so rather than
                                // letting item.cast<Format>() report a cast error about a
                                // type mismatch the caller did not make.
                                if (py::isinstance<Image>(item))
                                {
                                    raise_error(err_resource(
                                        "to render into images you already own, drop width and height: "
                                        "bz.RenderTarget(ctx, color=[image]) — the size, layers and mip "
                                        "levels come off the images"));
                                }
                                colors.push_back(item.cast<Format>());
                            }
                        }
                        else if (py::isinstance<Image>(color))
                        {
                            raise_error(err_resource(
                                "to render into an image you already own, drop width and height: "
                                "bz.RenderTarget(ctx, color=[image]) — the size, layers and mip "
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
                            context,
                            width,
                            height,
                            std::move(colors),
                            depth_format,
                            samples,
                            layers,
                            cube,
                            mip_levels,
                            name),
                        context.logger().get());
                }),
            py::arg("context"),
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
        // A target on images from create_image, instead of attachments the target
        // allocates. A second __init__ rather than optional width/height on the one
        // above: this signature has no width, height, samples, layers, cube or
        // mip_levels, because every one of those is a property of the images now.
        // pybind picks between the two on arity — width and height are required
        // positionals up there and absent here.
        .def(
            py::init(
                [](Context& context, py::object color, py::object depth, const std::string& name)
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
                                        "bz.RenderTarget(ctx, 512, 512, color=bz.Format.RGBA8) — or pass "
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

                    // The cross-Context guard belongs in the binding layer, as it does
                    // for every other resource: this catches a user mistake, not a C++
                    // invariant, and the GIL is held here.
                    for (const auto& image : colors)
                    {
                        require_same_context(&context, image->owner(), "RenderTarget(color=)");
                    }
                    if (depth_image)
                    {
                        require_same_context(&context, depth_image->owner(), "RenderTarget(depth=)");
                    }

                    return unwrap(
                        OffscreenTarget::create_from_images(context, std::move(colors), std::move(depth_image), name),
                        context.logger().get());
                }),
            py::arg("context"),
            py::kw_only(),
            py::arg("color") = py::none(),
            py::arg("depth") = py::none(),
            py::arg("name") = "")
        .def_property_readonly("width", [](const OffscreenTarget& t) { return t.extent().width; })
        .def_property_readonly("height", [](const OffscreenTarget& t) { return t.extent().height; })
        // The attachments are ordinary Images — this is the whole
        // render-to-texture API: target.color[0] / target.depth into set_image.
        .def_property_readonly(
            "color",
            [](const OffscreenTarget& t)
            {
                py::tuple out(t.colors().size());
                for (size_t i = 0; i < t.colors().size(); ++i)
                {
                    out[i] = py::cast(t.colors()[i]);
                }
                return out;
            })
        .def_property_readonly(
            "depth",
            [](const OffscreenTarget& t) -> py::object { return t.depth() ? py::cast(t.depth()) : py::none(); })
        // Render-to-layer / render-to-mip: a lightweight view of one subresource.
        // Pass it straight to cmd.rendering(...). Cube face i == layer i, Vulkan
        // order +X, -X, +Y, -Y, +Z, -Z.
        .def(
            "layer",
            [](std::shared_ptr<OffscreenTarget> self, std::uint32_t index, std::uint32_t mip)
            { return unwrap(self->layer(index, mip), nullptr); },
            py::arg("index"),
            py::arg("mip") = 0)
        // Multiview: one pass into every layer (the shader uses gl_ViewIndex).
        .def("all_layers", [](std::shared_ptr<OffscreenTarget> self) { return unwrap(self->all_layers(), nullptr); });

    // ── SwapchainRenderer ──
    // Inherits RenderTarget: presenting to a window is one way to consume a
    // rendered image, not the definition of rendering.
    py::enum_<PresentMode>(m, "PresentMode")
        .value("FIFO", PresentMode::FIFO)
        .value("MAILBOX", PresentMode::MAILBOX)
        .value("IMMEDIATE", PresentMode::IMMEDIATE)
        .value("FIFO_RELAXED", PresentMode::FIFO_RELAXED);

    py::class_<SwapchainRenderer, RenderTarget, std::shared_ptr<SwapchainRenderer>>(m, "SwapchainRenderer")
        .def(
            py::init(
                [](Window& window,
                   std::shared_ptr<Context> context,
                   PresentMode present_mode,
                   std::uint32_t samples,
                   bool stencil)
                {
                    auto sp = window.get_surface_provider();
                    return std::shared_ptr<SwapchainRenderer>(unwrap(
                        SwapchainRenderer::create(context, std::move(sp), present_mode, samples, stencil),
                        context->logger().get()));
                }),
            py::arg("window"),
            py::arg("context"),
            py::arg("present_mode") = PresentMode::MAILBOX,
            py::arg("samples") = 1,
            py::arg("stencil") = false,
            // The SurfaceProvider that get_surface_provider() returns captures the raw
            // GLFWwindow* and a pointer to the Window's own resize flag, and the renderer
            // keeps it for its whole life. So the Window has to outlive the renderer, and
            // nothing else says so: `del window` alone would leave both pointers dangling.
            // The hwnd overload below captures an integer and needs no such tie.
            py::keep_alive<1, 2>())
        .def(
            py::init(
                [](uint64_t hwnd,
                   std::shared_ptr<Context> context,
                   PresentMode present_mode,
                   std::uint32_t samples,
                   bool stencil) -> std::shared_ptr<SwapchainRenderer>
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
                        SwapchainRenderer::create(context, std::move(sp), present_mode, samples, stencil),
                        context->logger().get()));
#else
            raise_error(err_window("win32_hwnd constructor is only supported on Windows"));
#endif
                }),
            py::arg("win32_hwnd"),
            py::arg("context"),
            py::arg("present_mode") = PresentMode::MAILBOX,
            py::arg("samples") = 1,
            py::arg("stencil") = false)
        .def_property_readonly("present_mode", &SwapchainRenderer::present_mode)
        // A verb, not a settable property, because the request is a preference:
        // read present_mode back to see what the driver actually gave you.
        .def(
            "set_present_mode",
            [](SwapchainRenderer& self, PresentMode mode)
            {
                std::expected<void, Error> result;
                {
                    // Recreation takes the device idle — release the GIL for the
                    // wait, and unwrap only once it is back (raising needs it).
                    py::gil_scoped_release release;
                    result = self.set_present_mode(mode);
                }
                unwrap(std::move(result), nullptr);
            },
            py::arg("mode"))
        // bool, not a Frame: the frame belongs to the Context (ctx.begin_frame),
        // and what a window contributes to it is one acquired image. False means
        // "this window sits this frame out" — the others carry on.
        .def(
            "acquire",
            [](SwapchainRenderer& self) -> bool
            {
                std::expected<bool, Error> acquired;
                {
                    // Waits on the in-flight fence — release the GIL meanwhile.
                    py::gil_scoped_release release;
                    acquired = self.acquire();
                }
                return unwrap(std::move(acquired), self.context()->logger().get());
            })
        .def(
            "present",
            [](SwapchainRenderer& self, std::shared_ptr<CommandBuffer> cmd, bool capture)
            {
                require_same_context(self.owner(), cmd->owner(), "present");
                std::expected<void, Error> r;
                {
                    // May CPU-wait for an upload still decoding — release the GIL.
                    py::gil_scoped_release release;
                    r = present_command_buffer(self, std::move(cmd), capture);
                }
                unwrap(std::move(r), nullptr);
            },
            py::arg("cmd"),
            py::kw_only(),
            py::arg("capture") = false)
        // float milliseconds, or None until the ring has cycled once / on
        // devices without timestamp support. Per renderer, because the timestamp
        // pool is: two windows have two GPU frame times.
        .def_property_readonly(
            "gpu_time_ms",
            [](const SwapchainRenderer& r) -> py::object
            {
                auto ms = r.gpu_time_ms();
                return ms ? py::cast(*ms) : py::none();
            })
        // The frame asked for with present(capture=True). RGBA8 whatever channel
        // order the compositor picked, so out[y, x, 0] is red on every machine.
        // Shaped from the CAPTURE extent, not the current one: the window may
        // have been resized since.
        .def(
            "read_pixels",
            [](SwapchainRenderer& self) -> py::array
            {
                std::expected<std::vector<std::byte>, Error> bytes;
                {
                    // Waits on the capturing frame's fence.
                    py::gil_scoped_release release;
                    bytes = self.read_pixels();
                }
                auto data = unwrap(std::move(bytes), nullptr);
                const std::vector<py::ssize_t> shape{
                    static_cast<py::ssize_t>(self.capture_extent().height),
                    static_cast<py::ssize_t>(self.capture_extent().width),
                    4};
                py::array out(py::dtype("uint8"), shape);
                std::memcpy(out.mutable_data(), data.data(), data.size());
                return out;
            })
        .def_property_readonly("width", [](const SwapchainRenderer& r) { return r.extent().width; })
        .def_property_readonly("height", [](const SwapchainRenderer& r) { return r.extent().height; });

    // ── Key Constants ──
    m.attr("KEY_SPACE") = GLFW_KEY_SPACE;
    m.attr("KEY_APOSTROPHE") = GLFW_KEY_APOSTROPHE;
    m.attr("KEY_COMMA") = GLFW_KEY_COMMA;
    m.attr("KEY_MINUS") = GLFW_KEY_MINUS;
    m.attr("KEY_PERIOD") = GLFW_KEY_PERIOD;
    m.attr("KEY_SLASH") = GLFW_KEY_SLASH;
    m.attr("KEY_0") = GLFW_KEY_0;
    m.attr("KEY_1") = GLFW_KEY_1;
    m.attr("KEY_2") = GLFW_KEY_2;
    m.attr("KEY_3") = GLFW_KEY_3;
    m.attr("KEY_4") = GLFW_KEY_4;
    m.attr("KEY_5") = GLFW_KEY_5;
    m.attr("KEY_6") = GLFW_KEY_6;
    m.attr("KEY_7") = GLFW_KEY_7;
    m.attr("KEY_8") = GLFW_KEY_8;
    m.attr("KEY_9") = GLFW_KEY_9;
    m.attr("KEY_SEMICOLON") = GLFW_KEY_SEMICOLON;
    m.attr("KEY_EQUAL") = GLFW_KEY_EQUAL;
    m.attr("KEY_A") = GLFW_KEY_A;
    m.attr("KEY_B") = GLFW_KEY_B;
    m.attr("KEY_C") = GLFW_KEY_C;
    m.attr("KEY_D") = GLFW_KEY_D;
    m.attr("KEY_E") = GLFW_KEY_E;
    m.attr("KEY_F") = GLFW_KEY_F;
    m.attr("KEY_G") = GLFW_KEY_G;
    m.attr("KEY_H") = GLFW_KEY_H;
    m.attr("KEY_I") = GLFW_KEY_I;
    m.attr("KEY_J") = GLFW_KEY_J;
    m.attr("KEY_K") = GLFW_KEY_K;
    m.attr("KEY_L") = GLFW_KEY_L;
    m.attr("KEY_M") = GLFW_KEY_M;
    m.attr("KEY_N") = GLFW_KEY_N;
    m.attr("KEY_O") = GLFW_KEY_O;
    m.attr("KEY_P") = GLFW_KEY_P;
    m.attr("KEY_Q") = GLFW_KEY_Q;
    m.attr("KEY_R") = GLFW_KEY_R;
    m.attr("KEY_S") = GLFW_KEY_S;
    m.attr("KEY_T") = GLFW_KEY_T;
    m.attr("KEY_U") = GLFW_KEY_U;
    m.attr("KEY_V") = GLFW_KEY_V;
    m.attr("KEY_W") = GLFW_KEY_W;
    m.attr("KEY_X") = GLFW_KEY_X;
    m.attr("KEY_Y") = GLFW_KEY_Y;
    m.attr("KEY_Z") = GLFW_KEY_Z;
    m.attr("KEY_LEFT_BRACKET") = GLFW_KEY_LEFT_BRACKET;
    m.attr("KEY_BACKSLASH") = GLFW_KEY_BACKSLASH;
    m.attr("KEY_RIGHT_BRACKET") = GLFW_KEY_RIGHT_BRACKET;
    m.attr("KEY_GRAVE_ACCENT") = GLFW_KEY_GRAVE_ACCENT;
    m.attr("KEY_WORLD_1") = GLFW_KEY_WORLD_1;
    m.attr("KEY_WORLD_2") = GLFW_KEY_WORLD_2;
    m.attr("KEY_ESCAPE") = GLFW_KEY_ESCAPE;
    m.attr("KEY_ENTER") = GLFW_KEY_ENTER;
    m.attr("KEY_TAB") = GLFW_KEY_TAB;
    m.attr("KEY_BACKSPACE") = GLFW_KEY_BACKSPACE;
    m.attr("KEY_INSERT") = GLFW_KEY_INSERT;
    m.attr("KEY_DELETE") = GLFW_KEY_DELETE;
    m.attr("KEY_RIGHT") = GLFW_KEY_RIGHT;
    m.attr("KEY_LEFT") = GLFW_KEY_LEFT;
    m.attr("KEY_DOWN") = GLFW_KEY_DOWN;
    m.attr("KEY_UP") = GLFW_KEY_UP;
    m.attr("KEY_PAGE_UP") = GLFW_KEY_PAGE_UP;
    m.attr("KEY_PAGE_DOWN") = GLFW_KEY_PAGE_DOWN;
    m.attr("KEY_HOME") = GLFW_KEY_HOME;
    m.attr("KEY_END") = GLFW_KEY_END;
    m.attr("KEY_CAPS_LOCK") = GLFW_KEY_CAPS_LOCK;
    m.attr("KEY_SCROLL_LOCK") = GLFW_KEY_SCROLL_LOCK;
    m.attr("KEY_NUM_LOCK") = GLFW_KEY_NUM_LOCK;
    m.attr("KEY_PRINT_SCREEN") = GLFW_KEY_PRINT_SCREEN;
    m.attr("KEY_PAUSE") = GLFW_KEY_PAUSE;
    m.attr("KEY_F1") = GLFW_KEY_F1;
    m.attr("KEY_F2") = GLFW_KEY_F2;
    m.attr("KEY_F3") = GLFW_KEY_F3;
    m.attr("KEY_F4") = GLFW_KEY_F4;
    m.attr("KEY_F5") = GLFW_KEY_F5;
    m.attr("KEY_F6") = GLFW_KEY_F6;
    m.attr("KEY_F7") = GLFW_KEY_F7;
    m.attr("KEY_F8") = GLFW_KEY_F8;
    m.attr("KEY_F9") = GLFW_KEY_F9;
    m.attr("KEY_F10") = GLFW_KEY_F10;
    m.attr("KEY_F11") = GLFW_KEY_F11;
    m.attr("KEY_F12") = GLFW_KEY_F12;
    m.attr("KEY_F13") = GLFW_KEY_F13;
    m.attr("KEY_F14") = GLFW_KEY_F14;
    m.attr("KEY_F15") = GLFW_KEY_F15;
    m.attr("KEY_F16") = GLFW_KEY_F16;
    m.attr("KEY_F17") = GLFW_KEY_F17;
    m.attr("KEY_F18") = GLFW_KEY_F18;
    m.attr("KEY_F19") = GLFW_KEY_F19;
    m.attr("KEY_F20") = GLFW_KEY_F20;
    m.attr("KEY_F21") = GLFW_KEY_F21;
    m.attr("KEY_F22") = GLFW_KEY_F22;
    m.attr("KEY_F23") = GLFW_KEY_F23;
    m.attr("KEY_F24") = GLFW_KEY_F24;
    m.attr("KEY_F25") = GLFW_KEY_F25;
    m.attr("KEY_KP_0") = GLFW_KEY_KP_0;
    m.attr("KEY_KP_1") = GLFW_KEY_KP_1;
    m.attr("KEY_KP_2") = GLFW_KEY_KP_2;
    m.attr("KEY_KP_3") = GLFW_KEY_KP_3;
    m.attr("KEY_KP_4") = GLFW_KEY_KP_4;
    m.attr("KEY_KP_5") = GLFW_KEY_KP_5;
    m.attr("KEY_KP_6") = GLFW_KEY_KP_6;
    m.attr("KEY_KP_7") = GLFW_KEY_KP_7;
    m.attr("KEY_KP_8") = GLFW_KEY_KP_8;
    m.attr("KEY_KP_9") = GLFW_KEY_KP_9;
    m.attr("KEY_KP_DECIMAL") = GLFW_KEY_KP_DECIMAL;
    m.attr("KEY_KP_DIVIDE") = GLFW_KEY_KP_DIVIDE;
    m.attr("KEY_KP_MULTIPLY") = GLFW_KEY_KP_MULTIPLY;
    m.attr("KEY_KP_SUBTRACT") = GLFW_KEY_KP_SUBTRACT;
    m.attr("KEY_KP_ADD") = GLFW_KEY_KP_ADD;
    m.attr("KEY_KP_ENTER") = GLFW_KEY_KP_ENTER;
    m.attr("KEY_KP_EQUAL") = GLFW_KEY_KP_EQUAL;
    m.attr("KEY_LEFT_SHIFT") = GLFW_KEY_LEFT_SHIFT;
    m.attr("KEY_LEFT_CONTROL") = GLFW_KEY_LEFT_CONTROL;
    m.attr("KEY_LEFT_ALT") = GLFW_KEY_LEFT_ALT;
    m.attr("KEY_LEFT_SUPER") = GLFW_KEY_LEFT_SUPER;
    m.attr("KEY_RIGHT_SHIFT") = GLFW_KEY_RIGHT_SHIFT;
    m.attr("KEY_RIGHT_CONTROL") = GLFW_KEY_RIGHT_CONTROL;
    m.attr("KEY_RIGHT_ALT") = GLFW_KEY_RIGHT_ALT;
    m.attr("KEY_RIGHT_SUPER") = GLFW_KEY_RIGHT_SUPER;
    m.attr("KEY_MENU") = GLFW_KEY_MENU;
    m.attr("KEY_LAST") = GLFW_KEY_LAST;

    m.attr("MOUSE_BUTTON_LEFT") = GLFW_MOUSE_BUTTON_LEFT;
    m.attr("MOUSE_BUTTON_RIGHT") = GLFW_MOUSE_BUTTON_RIGHT;
    m.attr("MOUSE_BUTTON_MIDDLE") = GLFW_MOUSE_BUTTON_MIDDLE;

    m.attr("CURSOR_NORMAL") = GLFW_CURSOR_NORMAL;
    m.attr("CURSOR_DISABLED") = GLFW_CURSOR_DISABLED;
    m.attr("CURSOR_HIDDEN") = GLFW_CURSOR_HIDDEN;
}