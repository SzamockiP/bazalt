#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <atomic>
#include <cstddef>
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
void context_submit(Context& context, std::shared_ptr<CommandBuffer> cmd);

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
        raise_error(err_resource("Could not infer data type from list elements"));
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
                    "create_image: 3-channel images have no portable GPU format; "
                    "pad to 4 channels first, e.g. "
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
    VkCommandBuffer record_frame(CommandBuffer& cmd, std::uint32_t frame_index, TimestampRange ts = {})
    {
        VkCommandBuffer vkCmd = cmd.get(frame_index);
        vkResetCommandBuffer(vkCmd, 0);

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};

        if (auto e = check(vkBeginCommandBuffer(vkCmd, &beginInfo), "begin recording command buffer"))
        {
            raise_error(*e);
        }

        // The queries must be reset on the device before use; doing it here (rather
        // than once up front) keeps them per-frame and needs no hostQueryReset.
        if (ts.pool != VK_NULL_HANDLE)
        {
            vkCmdResetQueryPool(vkCmd, ts.pool, ts.first, 2);
            vkCmdWriteTimestamp(vkCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts.pool, ts.first);
        }

        cmd.execute(vkCmd, FrameContext{frame_index});

        if (ts.pool != VK_NULL_HANDLE)
        {
            vkCmdWriteTimestamp(vkCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts.pool, ts.first + 1);
        }

        if (auto e = check(vkEndCommandBuffer(vkCmd), "record command buffer"))
        {
            raise_error(*e);
        }

        return vkCmd;
    }

} // namespace
void SwapchainRenderer::submit(std::shared_ptr<CommandBuffer> cmd, std::uint64_t upload_wait_serial)
{
    TimestampRange ts{};
    if (timestamps_supported())
    {
        ts = {timestamp_pool(), 2 * current_frame()};
    }
    end_frame(record_frame(*cmd, current_frame(), ts), upload_wait_serial);
    if (timestamps_supported())
    {
        // The slot now holds results begin_frame can read once its fence signals.
        mark_timestamp_written();
    }
}

// Upload residency, per command buffer: every image this recording references
// must at least be *submitted* (CPU-wait while the worker is still decoding —
// correctness first), and the frame's GPU work then waits on the submission
// timeline for the highest upload it depends on (zero CPU stall in the steady
// state). Images with no pending upload — RTT attachments, synchronous
// uploads — short-circuit to 0.
std::uint64_t require_uploads_resident(CommandBuffer& cmd)
{
    std::uint64_t wait_serial = 0;
    for (const auto& set : cmd.used_sets())
    {
        for (const auto& bi : set->images())
        {
            auto serial = bi.image->require_resident();
            if (!serial)
            {
                raise_error(serial.error());
            }
            wait_serial = (std::max)(wait_serial, *serial);
        }
    }
    return wait_serial;
}

std::expected<void, Error> Frame::submit(std::shared_ptr<CommandBuffer> cmd)
{
    if (submitted)
    {
        return std::unexpected(
            err_resource("This Frame was already submitted. Call begin_frame() again to get the next one."));
    }
    if (serial != renderer->current_serial())
    {
        return std::unexpected(err_resource(
            "This Frame is stale: begin_frame() has run again since it was acquired. "
            "Acquire, submit and drop a Frame within one tick."));
    }
    const std::uint64_t upload_wait_serial = require_uploads_resident(*cmd);
    renderer->submit(std::move(cmd), upload_wait_serial);
    submitted = true;
    return {};
}

// The state behind `with cmd.rendering(target, ...):` — carries what
// __enter__/__exit__ need to record the begin/end pair. Deliberately a plain
// struct bound only for its dunder methods; begin_rendering/end_rendering
// stay public, this is sugar, not a replacement.
struct RenderingScope
{
    std::shared_ptr<CommandBuffer> cmd;
    std::shared_ptr<RenderTarget> target;
    std::vector<float> clear_color;
};

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

// Readback shaped for numpy: (h, w, channels) — or (h, w) for single-channel
// formats — with the dtype the format table dictates. Shared by Image.read and
// RenderTarget.read_pixels.
py::array image_to_numpy(Image& image)
{
    auto bytes = unwrap(image.read(), nullptr);
    const FormatInfo info = format_info(image.format());

    std::vector<py::ssize_t> shape;
    if (info.channels == 1)
    {
        shape = {static_cast<py::ssize_t>(image.height()), static_cast<py::ssize_t>(image.width())};
    }
    else
    {
        shape = {
            static_cast<py::ssize_t>(image.height()),
            static_cast<py::ssize_t>(image.width()),
            static_cast<py::ssize_t>(info.channels)};
    }

    py::array out(py::dtype(info.numpy_dtype), shape);
    std::memcpy(out.mutable_data(), bytes.data(), bytes.size());
    return out;
}

void context_submit(Context& context, std::shared_ptr<CommandBuffer> cmd)
{
    // Drain hot reloads BEFORE recording, so an edit-then-submit picks up the new
    // pipeline in THIS submit. The headless path advances the ring only after
    // submitting, so hooking the drain there (as the windowed path does) would
    // delay every reload by one submit — and the whole test suite is headless.
    if (auto* hr = context.hot_reload())
    {
        hr->drain();
    }

    const std::uint64_t upload_wait_serial = require_uploads_resident(*cmd);
    VkCommandBuffer vkCmd = record_frame(*cmd, context.frame_index());

    VkSemaphore timeline = context.submit_timeline();
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    {
        std::lock_guard lock(context.queue_mutex());
        const std::uint64_t serial = context.advance_submit_serial();

        VkTimelineSemaphoreSubmitInfo timelineInfo{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &upload_wait_serial,
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

        if (auto e =
                check(vkQueueSubmit(context.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE), "submit command buffer"))
        {
            raise_error(*e);
        }

        // Blocking. There is no swapchain to pace against here; the timeline
        // exists for uploads and deferred destruction, not (yet) for making
        // headless submits asynchronous.
        if (auto e = check(vkQueueWaitIdle(context.graphics_queue()), "wait for submitted command buffer"))
        {
            raise_error(*e);
        }
    }

    // The wait-idle above proves everything through the current serial is done.
    context.flush_deletion_queue();

    // A headless submit is a frame too: advance the ring so DynamicBuffer
    // slots and frame descriptor sets rotate (this path used to sit on slot 0
    // forever). After, not before, submitting — an update() made before this
    // call must land in the slot this submit reads.
    context.advance_frame();
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
        .value("ERROR", Severity::Error)
        .export_values();

    py::enum_<Source>(m, "Source")
        .value("GENERAL", Source::General)
        .value("VALIDATION", Source::Validation)
        .value("WINDOW", Source::Window)
        .value("SHADER", Source::Shader)
        .value("UPLOAD", Source::Upload)
        .value("DEVICE", Source::Device)
        .export_values();

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
        .export_values();

    py::enum_<BufferType>(m, "BufferType")
        .value("VERTEX", BufferType::VERTEX)
        .value("INDEX", BufferType::INDEX)
        .value("UNIFORM", BufferType::UNIFORM)
        .value("STORAGE", BufferType::STORAGE)
        .export_values();

    py::enum_<DataType>(m, "DataType")
        .value("FLOAT", DataType::FLOAT)
        .value("UINT32", DataType::UINT32)
        .value("UINT16", DataType::UINT16)
        .value("INT32", DataType::INT32)
        .export_values();

    py::enum_<ShaderStage>(m, "ShaderStage")
        .value("VERTEX", ShaderStage::VERTEX)
        .value("FRAGMENT", ShaderStage::FRAGMENT)
        .value("COMPUTE", ShaderStage::COMPUTE)
        .export_values();

    py::enum_<VertexFormat>(m, "VertexFormat")
        .value("FLOAT2", VertexFormat::FLOAT2)
        .value("FLOAT3", VertexFormat::FLOAT3)
        .value("FLOAT4", VertexFormat::FLOAT4)
        .export_values();

    py::enum_<Topology>(m, "Topology")
        .value("TRIANGLE_LIST", Topology::TRIANGLE_LIST)
        .value("POINT_LIST", Topology::POINT_LIST)
        .value("LINE_LIST", Topology::LINE_LIST)
        .export_values();

    // The vocabulary of cmd.barrier() in manual mode (auto_barriers=False).
    py::enum_<Access>(m, "Access")
        .value("SHADER_READ", Access::SHADER_READ)
        .value("SHADER_WRITE", Access::SHADER_WRITE)
        .value("VERTEX_READ", Access::VERTEX_READ)
        .value("INDEX_READ", Access::INDEX_READ)
        .value("UNIFORM_READ", Access::UNIFORM_READ)
        .export_values();

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
        .value("D32F", Format::D32F);

    py::enum_<Filter>(m, "Filter").value("LINEAR", Filter::LINEAR).value("NEAREST", Filter::NEAREST);

    py::enum_<AddressMode>(m, "AddressMode")
        .value("REPEAT", AddressMode::REPEAT)
        .value("CLAMP", AddressMode::CLAMP)
        .value("MIRROR", AddressMode::MIRROR);

    py::enum_<CompareOp>(m, "CompareOp")
        .value("NEVER", CompareOp::NEVER)
        .value("LESS", CompareOp::LESS)
        .value("EQUAL", CompareOp::EQUAL)
        .value("LESS_OR_EQUAL", CompareOp::LESS_OR_EQUAL)
        .value("GREATER", CompareOp::GREATER)
        .value("NOT_EQUAL", CompareOp::NOT_EQUAL)
        .value("GREATER_OR_EQUAL", CompareOp::GREATER_OR_EQUAL)
        .value("ALWAYS", CompareOp::ALWAYS);

    py::enum_<CullMode>(m, "CullMode")
        .value("NONE", CullMode::NONE)
        .value("BACK", CullMode::BACK)
        .value("FRONT", CullMode::FRONT)
        .value("FRONT_AND_BACK", CullMode::FRONT_AND_BACK)
        .export_values();

    py::enum_<FrontFace>(m, "FrontFace")
        .value("CLOCKWISE", FrontFace::CLOCKWISE)
        .value("COUNTER_CLOCKWISE", FrontFace::COUNTER_CLOCKWISE)
        .export_values();

    py::enum_<MemoryUsage>(m, "MemoryUsage")
        .value("STATIC", MemoryUsage::STATIC)
        .value("DYNAMIC", MemoryUsage::DYNAMIC)
        .export_values();

    py::class_<MouseState>(m, "MouseState").def_readonly("dx", &MouseState::dx).def_readonly("dy", &MouseState::dy);

    py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer")
        .def(
            "update",
            [](Buffer& buffer, std::string_view data)
            { unwrap(buffer.update(std::as_bytes(std::span(data.data(), data.size()))), nullptr); })
        .def(
            "update",
            [](Buffer& buffer, py::buffer b)
            {
                py::buffer_info info = b.request();
                const size_t nbytes = contiguous_nbytes(info, "Buffer.update");
                unwrap(buffer.update({static_cast<const std::byte*>(info.ptr), nbytes}), nullptr);
            })
        .def(
            "update",
            [](Buffer& buffer, py::list list, std::optional<DataType> dataType)
            {
                if (list.empty())
                    return;
                DataType actualType = resolve_data_type(list, dataType, DataType::INT32);
                with_list_bytes(
                    list,
                    actualType,
                    [&](const void* data, size_t nbytes)
                    { unwrap(buffer.update({static_cast<const std::byte*>(data), nbytes}), nullptr); });
            },
            py::arg("list"),
            py::arg("data_type") = py::none())
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
            py::arg("dtype"));

    py::class_<ShaderModule, std::shared_ptr<ShaderModule>>(m, "ShaderModule")
        .def_property_readonly("path", &ShaderModule::path)
        .def_property_readonly("includes", &ShaderModule::includes)
        .def_property_readonly(
            "spirv",
            [](const ShaderModule& self)
            {
                const auto& words = self.spirv();
                return py::bytes(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(uint32_t));
            });

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
        .def("read", [](Image& self) -> py::array { return image_to_numpy(self); });

    py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Sampler");

    py::class_<Pipeline, std::shared_ptr<Pipeline>>(m, "Pipeline");

    // Lambdas, not member pointers: the setters take a deducing-this object
    // parameter, so &GraphicsPipelineBuilder::vertex_shader would be a plain
    // function pointer that .def() cannot treat as a method.
    py::class_<GraphicsPipelineBuilder, std::shared_ptr<GraphicsPipelineBuilder>>(m, "GraphicsPipelineBuilder")
        .def(
            "vertex_shader",
            [](GraphicsPipelineBuilder& self, std::shared_ptr<ShaderModule> shader) -> GraphicsPipelineBuilder&
            { return self.vertex_shader(std::move(shader)); })
        .def(
            "fragment_shader",
            [](GraphicsPipelineBuilder& self, std::shared_ptr<ShaderModule> shader) -> GraphicsPipelineBuilder&
            { return self.fragment_shader(std::move(shader)); })
        .def(
            "vertex_format",
            [](GraphicsPipelineBuilder& self, const std::vector<VertexFormat>& formats) -> GraphicsPipelineBuilder&
            { return self.vertex_format(formats); })
        .def(
            "depth_test",
            [](GraphicsPipelineBuilder& self, bool enable) -> GraphicsPipelineBuilder&
            { return self.depth_test(enable); })
        .def(
            "cull_mode",
            [](GraphicsPipelineBuilder& self, CullMode mode, FrontFace frontFace) -> GraphicsPipelineBuilder&
            { return self.cull_mode(mode, frontFace); })
        .def(
            "blend",
            [](GraphicsPipelineBuilder& self, bool enable) -> GraphicsPipelineBuilder& { return self.blend(enable); })
        .def(
            "topology",
            [](GraphicsPipelineBuilder& self, Topology topology) -> GraphicsPipelineBuilder&
            { return self.topology(topology); })
        .def(
            "push_constant",
            [](GraphicsPipelineBuilder& self, uint32_t size, ShaderStage stage) -> GraphicsPipelineBuilder&
            { return self.push_constant(size, stage); })
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
            { return self.shader(std::move(shader)); })
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
            { unwrap(self.set_image(binding, std::move(image), std::move(sampler)), nullptr); },
            py::arg("binding"),
            py::arg("image"),
            py::arg("sampler") = py::none())
        .def(
            "set_storage_image",
            [](DescriptorSet& self, uint32_t binding, std::shared_ptr<Image> image)
            { unwrap(self.set_storage_image(binding, std::move(image)), nullptr); },
            py::arg("binding"),
            py::arg("image"))
        .def(
            "set_buffer",
            [](DescriptorSet& self, uint32_t binding, std::shared_ptr<Buffer> buffer)
            { unwrap(self.set_buffer(binding, std::move(buffer)), nullptr); },
            py::arg("binding"),
            py::arg("buffer"));

    py::class_<DescriptorPool, std::shared_ptr<DescriptorPool>>(m, "DescriptorPool")
        .def(
            "allocate_set",
            [](DescriptorPool& pool, std::shared_ptr<Pipeline> pipeline, uint32_t setIndex) -> py::object
            { return py::cast(unwrap(pool.allocate_descriptor_set(pipeline, setIndex), pool.logger().get())); },
            py::arg("pipeline"),
            py::arg("set"))
        .def(
            "allocate_frame_set",
            [](DescriptorPool& pool, std::shared_ptr<Pipeline> pipeline, uint32_t setIndex) -> py::object
            { return py::cast(unwrap(pool.allocate_frame_descriptor_set(pipeline, setIndex), pool.logger().get())); },
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
               const std::vector<float>& clear_color)
            {
                self->begin_rendering(std::move(target), clear_color);
                return self;
            },
            py::arg("target"),
            py::arg("clear_color") = std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f})
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
               const std::vector<float>& clear_color)
            { return RenderingScope{std::move(self), std::move(target), clear_color}; },
            py::arg("target"),
            py::arg("clear_color") = std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f})
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
                self->bind_pipeline(std::move(pipeline));
                return self;
            },
            py::arg("pipeline"))
        .def(
            "bind_vertex_buffer",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Buffer> buffer)
            {
                self->bind_vertex_buffer(std::move(buffer));
                return self;
            },
            py::arg("buffer"))
        .def(
            "bind_index_buffer",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Buffer> buffer)
            {
                self->bind_index_buffer(std::move(buffer));
                return self;
            },
            py::arg("buffer"))
        .def(
            "draw",
            [](std::shared_ptr<CommandBuffer> self, uint32_t vertex_count)
            {
                self->draw(vertex_count);
                return self;
            },
            py::arg("vertex_count"))
        .def(
            "draw_indexed",
            [](std::shared_ptr<CommandBuffer> self, uint32_t index_count, uint32_t first_index, int32_t vertex_offset)
            {
                self->draw_indexed(index_count, first_index, vertex_offset);
                return self;
            },
            py::arg("index_count"),
            py::arg("first_index") = 0,
            py::arg("vertex_offset") = 0)
        .def(
            "draw_indexed_instanced",
            [](std::shared_ptr<CommandBuffer> self,
               uint32_t index_count,
               uint32_t instance_count,
               uint32_t first_index,
               int32_t vertex_offset)
            {
                self->draw_indexed_instanced(index_count, instance_count, first_index, vertex_offset);
                return self;
            },
            py::arg("index_count"),
            py::arg("instance_count"),
            py::arg("first_index") = 0,
            py::arg("vertex_offset") = 0)
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
        .def(
            "barrier",
            [](std::shared_ptr<CommandBuffer> self, std::shared_ptr<Buffer> buffer, Access src, Access dst)
            {
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
                unwrap(self->generate_mipmaps(std::move(image), src), nullptr);
                return self;
            },
            py::arg("image"),
            py::kw_only(),
            py::arg("src") = Access::SHADER_READ)
        // No stage argument: the Pipeline already records which stages its push
        // constant range covers, so repeating it could only ever be wrong.
        .def(
            "push_constants",
            [](std::shared_ptr<CommandBuffer> self,
               std::shared_ptr<Pipeline> pipeline,
               uint32_t offset,
               std::string_view data)
            {
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
                self.cmd->begin_rendering(self.target, self.clear_color);
                return self.cmd;
            })
        .def(
            "__exit__",
            [](RenderingScope& self, py::object, py::object, py::object)
            {
                self.cmd->end_rendering(self.target);
                return false; // never swallow exceptions
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
        .def_property_readonly(
            "ms", [](const Timer& self) { return self.cmd->read_timer(self.index, self.generation); });

    // ── Window (GLFW) ──
    py::class_<Window>(m, "Window")
        .def(
            py::init(
                [](int width, int height, const std::string& title, std::shared_ptr<Logger> logger)
                {
                    // Window used to have no way to reach a Logger at all, so GLFW's own
                    // diagnostics went nowhere.
                    return unwrap(Window::create(width, height, title, logger), logger.get());
                }),
            py::arg("width"),
            py::arg("height"),
            py::arg("title"),
            py::arg("logger") = py::none())
        .def("is_open", &Window::is_open)
        .def("should_close", &Window::should_close)
        .def("poll_events", &Window::poll_events)
        .def("is_key_pressed", &Window::is_key_pressed, py::arg("key"))
        .def("is_mouse_button_pressed", &Window::is_mouse_button_pressed, py::arg("button"))
        .def("set_cursor_mode", &Window::set_cursor_mode, py::arg("mode"))
        .def("get_mouse_state", &Window::get_mouse_state)
        .def("set_title", &Window::set_title, py::arg("title"))
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
    py::class_<Context, std::shared_ptr<Context>>(m, "Context")
        .def(
            py::init(
                [](std::shared_ptr<Logger> logger,
                   const std::string& validation,
                   std::vector<Feature> features,
                   std::vector<Feature> optional,
                   std::uint32_t frames_in_flight,
                   std::vector<std::string> raw_extensions,
                   bool auto_barriers,
                   bool hot_reload,
                   bool gpu_timing)
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
                    config.raw_extensions = std::move(raw_extensions);
                    config.auto_barriers = auto_barriers;
                    config.gpu_timing = gpu_timing;

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
            py::arg("raw_extensions") = std::vector<std::string>{},
            py::arg("auto_barriers") = true,
            py::arg("hot_reload") = false,
            py::arg("gpu_timing") = false)
        .def_property_readonly("auto_barriers", &Context::auto_barriers)
        .def_property_readonly("frames_in_flight", &Context::frames_in_flight)
        .def_property_readonly("logger", &Context::logger)
        .def("supports", &Context::supports, py::arg("feature"))
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
            [](Context& self, const std::string& path, ShaderStage stage, std::optional<std::string> source)
                -> py::object
            {
                // Only file-backed shaders are watchable: a source= virtual name may
                // not exist on disk, and .spv is recompiled from its own path too.
                const bool from_file = !source.has_value();
                auto module =
                    unwrap(ShaderCompiler::compile(self, path, stage, std::move(source)), self.logger().get());
                if (auto* hr = self.hot_reload(); hr && from_file && std::filesystem::exists(path))
                {
                    hr->watch_shader(module);
                }
                return py::cast(module);
            },
            py::arg("path"),
            py::arg("stage"),
            py::kw_only(),
            py::arg("source") = py::none())
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
                // submit. img.ready / img.wait() / ctx.wait_for_uploads() are the
                // explicit-control verbs.
                if (!self.upload_manager())
                {
                    self.set_upload_manager(std::make_unique<UploadManager>(self));
                }
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
                if (!self.upload_manager())
                {
                    self.set_upload_manager(std::make_unique<UploadManager>(self));
                }
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
        .def_property_readonly(
            "uploads_done",
            [](const Context& self) { return self.upload_manager() ? self.upload_manager()->uploads_done() : true; })
        .def_property_readonly(
            "upload_progress",
            [](const Context& self)
            {
                // Progress of the current batch of load_image calls, 0.0 .. 1.0
                // (1.0 when idle) — a loading bar without user-side threads.
                return self.upload_manager() ? self.upload_manager()->upload_progress() : 1.0;
            })
        .def(
            "wait_for_uploads",
            [](Context& self)
            {
                if (self.upload_manager())
                {
                    py::gil_scoped_release release;
                    self.upload_manager()->wait_all();
                }
            })
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
                                "create_image(cube=True) implies 6 layers; drop layers= or pass layers=6, got {}",
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
                            std::format("create_image: every layer must share shape and dtype; layer {} differs", i)));
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
        .def(
            "create_sampler",
            [](Context& self,
               Filter filter,
               AddressMode address_mode,
               bool anisotropy,
               std::optional<CompareOp> compare) -> py::object
            {
                // Cached: identical descriptions return the identical object.
                return py::cast(unwrap(
                    self.get_sampler(SamplerDesc{filter, address_mode, anisotropy, compare}), self.logger().get()));
            },
            py::arg("filter") = Filter::LINEAR,
            py::arg("address_mode") = AddressMode::REPEAT,
            py::arg("anisotropy") = true,
            py::arg("compare") = py::none())
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
        // The headless counterpart of frame.submit(): no swapchain, no present.
        .def(
            "submit",
            [](Context& self, std::shared_ptr<CommandBuffer> cmd)
            {
                // Blocking (wait-idle inside) — release the GIL for the duration.
                py::gil_scoped_release release;
                context_submit(self, std::move(cmd));
            },
            py::arg("cmd"));

    // ── RenderTarget ──
    py::class_<RenderTarget, std::shared_ptr<RenderTarget>>(m, "RenderTargetBase");

    py::class_<OffscreenTarget, RenderTarget, std::shared_ptr<OffscreenTarget>>(m, "RenderTarget")
        // color: None | Format | [Format, ...]; depth: None | Format. A bool
        // depth is refused with a migration hint (it was one in 0.4).
        .def(
            py::init(
                [](Context& context, std::uint32_t width, std::uint32_t height, py::object color, py::object depth)
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
                                colors.push_back(item.cast<Format>());
                            }
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
                        OffscreenTarget::create(context, width, height, std::move(colors), depth_format),
                        context.logger().get());
                }),
            py::arg("context"),
            py::arg("width"),
            py::arg("height"),
            py::arg("color") = Format::RGBA8,
            py::arg("depth") = py::none())
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
        // Kept as the ergonomic spelling for colour 0; the general form is
        // target.color[i].read().
        .def(
            "read_pixels",
            [](OffscreenTarget& self) -> py::array
            {
                if (self.colors().empty())
                {
                    raise_error(err_resource("read_pixels() on a depth-only RenderTarget; read target.depth instead"));
                }
                return image_to_numpy(*self.colors()[0]);
            });

    // ── SwapchainRenderer ──
    // Inherits RenderTarget: presenting to a window is one way to consume a
    // rendered image, not the definition of rendering.
    py::enum_<PresentMode>(m, "PresentMode")
        .value("FIFO", PresentMode::FIFO)
        .value("MAILBOX", PresentMode::MAILBOX)
        .value("IMMEDIATE", PresentMode::IMMEDIATE);

    py::class_<SwapchainRenderer, RenderTarget, std::shared_ptr<SwapchainRenderer>>(m, "SwapchainRenderer")
        .def(
            py::init(
                [](Window& window, std::shared_ptr<Context> context, PresentMode present_mode)
                {
                    auto sp = window.get_surface_provider();
                    return std::shared_ptr<SwapchainRenderer>(unwrap(
                        SwapchainRenderer::create(context, std::move(sp), present_mode), context->logger().get()));
                }),
            py::arg("window"),
            py::arg("context"),
            py::arg("present_mode") = PresentMode::MAILBOX)
        .def(
            py::init(
                [](uint64_t hwnd, std::shared_ptr<Context> context, PresentMode present_mode)
                    -> std::shared_ptr<SwapchainRenderer>
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
                        SwapchainRenderer::create(context, std::move(sp), present_mode), context->logger().get()));
#else
            raise_error(err_window("win32_hwnd constructor is only supported on Windows"));
#endif
                }),
            py::arg("win32_hwnd"),
            py::arg("context"),
            py::arg("present_mode") = PresentMode::MAILBOX)
        .def_property_readonly("present_mode", &SwapchainRenderer::present_mode)
        // Frame | None instead of bool: "the frame exists" and "here is the
        // frame" are the same fact, so return it. renderer.submit is gone —
        // submitting lives on the Frame you were handed.
        .def(
            "begin_frame",
            [](std::shared_ptr<SwapchainRenderer> self) -> py::object
            {
                bool acquired;
                {
                    // Waits on the in-flight fence — release the GIL meanwhile.
                    py::gil_scoped_release release;
                    acquired = self->begin_frame();
                }
                if (!acquired)
                {
                    return py::none();
                }
                auto frame = std::make_shared<Frame>();
                frame->renderer = self;
                frame->serial = self->current_serial();
                frame->frame_index = self->current_frame();
                frame->image_index = self->current_image_index();
                // Snapshot the GPU time read back during begin_frame, so it stays
                // stable for the frame's lifetime regardless of later begin_frames.
                frame->gpu_time_ms = self->gpu_time_ms();
                return py::cast(frame);
            })
        .def_property_readonly("width", [](const SwapchainRenderer& r) { return r.extent().width; })
        .def_property_readonly("height", [](const SwapchainRenderer& r) { return r.extent().height; });

    py::class_<Frame, std::shared_ptr<Frame>>(m, "Frame")
        .def(
            "submit",
            [](Frame& self, std::shared_ptr<CommandBuffer> cmd)
            {
                std::expected<void, Error> r;
                {
                    // May CPU-wait for an upload still decoding — release the GIL.
                    py::gil_scoped_release release;
                    r = self.submit(std::move(cmd));
                }
                unwrap(std::move(r), nullptr);
            },
            py::arg("cmd"))
        .def_property_readonly("frame_index", [](const Frame& f) { return f.frame_index; })
        // float milliseconds, or None until the ring has cycled once / on
        // devices without timestamp support.
        .def_property_readonly(
            "gpu_time_ms",
            [](const Frame& f) -> py::object { return f.gpu_time_ms ? py::cast(*f.gpu_time_ms) : py::none(); });

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