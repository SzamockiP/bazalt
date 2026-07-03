#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <print>
#include <atomic>
#include <expected>

#include "Logger.hpp"
#include "window.hpp"
#include "Renderer.hpp"
#include "Buffer.hpp"
#include "ShaderCompiler.hpp"
#include "Pipeline.hpp"
#include "CommandBuffer.hpp"

namespace py = pybind11;

class Engine {
public:
    Engine() = default;

    ~Engine()
    {
        stop();
    }

    py::function on_error(py::function callback)
    {
        logger_.register_callback(callback);
        return callback;
    }

    void init(int width, int height, const std::string& title)
    {
        renderer_.reset();
        window_.reset();

        auto res = Window::create(width, height, title)
            .and_then([&](std::unique_ptr<Window> win) {
                window_ = std::move(win);
                return Renderer::create(logger_, *window_);
            })
            .and_then([&](std::unique_ptr<Renderer> ren) {
                renderer_ = std::move(ren);
                return std::expected<void, std::string>{};
            })
            .or_else([&](const std::string& err) -> std::expected<void, std::string> {
                logger_.log(err);
                throw std::runtime_error(err);
            });
    }

    void run(py::object app_instance = py::none())
    {
        is_running_.store(true, std::memory_order_relaxed);

        try
        {
            py::gil_scoped_release release;

            while (is_running_.load(std::memory_order_relaxed))
            {
                window_->poll_events();
                if (window_->should_close())
                {
                    stop();
                }

                {
                    py::gil_scoped_acquire acquire;
                    if (PyErr_CheckSignals() != 0)
                    {
                        throw py::error_already_set();
                    }
                    
                    renderer_->begin_frame();
                    
                    if (!app_instance.is_none()) {
                        frame_function_(app_instance);
                    } else {
                        frame_function_();
                    }
                }
            }
        }
        catch (...)
        {
            logger_.shutdown();
            throw;
        }
        
        if (renderer_) {
            vkDeviceWaitIdle(renderer_->device());
        }
        logger_.shutdown();
    }

    bool running() const
    {
        return is_running_.load(std::memory_order_relaxed);
    }

    void stop()
    {
        is_running_.store(false, std::memory_order_relaxed);
    }

    void log(const std::string& msg)
    {
        logger_.log(msg);
    }

    py::function on_frame(py::function fun)
    {
        frame_function_ = fun;
        return fun;
    }

    MouseState get_mouse_state() const
    {
        return window_->get_mouse_state();
    }

    bool is_key_pressed(int key) const
    {
        return window_->is_key_pressed(key);
    }

    py::object create_buffer(py::list list, BufferType type, DataType dataType) {
        size_t count = list.size();
        std::expected<std::shared_ptr<Buffer>, std::string> res = std::unexpected("Unknown data type");

        if (dataType == DataType::FLOAT) {
            std::vector<float> data(count);
            for(size_t i=0; i<count; ++i) data[i] = list[i].cast<float>();
            res = Buffer::create(*renderer_, data.data(), count * sizeof(float), type);
        } else if (dataType == DataType::UINT32) {
            std::vector<uint32_t> data(count);
            for(size_t i=0; i<count; ++i) data[i] = list[i].cast<uint32_t>();
            res = Buffer::create(*renderer_, data.data(), count * sizeof(uint32_t), type);
        } else if (dataType == DataType::UINT16) {
            std::vector<uint16_t> data(count);
            for(size_t i=0; i<count; ++i) data[i] = list[i].cast<uint16_t>();
            res = Buffer::create(*renderer_, data.data(), count * sizeof(uint16_t), type);
        } else if (dataType == DataType::INT32) {
            std::vector<int32_t> data(count);
            for(size_t i=0; i<count; ++i) data[i] = list[i].cast<int32_t>();
            res = Buffer::create(*renderer_, data.data(), count * sizeof(int32_t), type);
        }

        if (res) {
            return py::cast(res.value());
        } else {
            logger_.log(res.error());
            throw std::runtime_error(res.error());
        }
    }

    py::object create_command_buffer() {
        auto res = CommandBuffer::create(*renderer_);
        if (res) {
            return py::cast(res.value());
        } else {
            logger_.log(res.error());
            throw std::runtime_error(res.error());
        }
    }

    std::shared_ptr<PipelineBuilder> create_pipeline() {
        return std::make_shared<PipelineBuilder>(*renderer_);
    }

    py::object compile_shader(const std::string& path, ShaderStage stage) {
        auto res = ShaderCompiler::compile(renderer_->device(), path, stage);
        if (res) {
            return py::cast(res.value());
        } else {
            logger_.log(res.error());
            throw std::runtime_error(res.error());
        }
    }

    void submit(std::shared_ptr<CommandBuffer> cmd) {
        if (cmd->get() != VK_NULL_HANDLE) {
            vkEndCommandBuffer(cmd->get());
        }
        renderer_->end_frame(cmd->get());
    }

private:
    std::atomic<bool> is_running_{false};
    Logger logger_;
    std::jthread logic_thread_;

    std::unique_ptr<Window> window_;
    std::unique_ptr<Renderer> renderer_;

    py::function frame_function_;
};

PYBIND11_MODULE(lumapy, m) {
    m.doc() = "LumaPy module";

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
        .export_values();

    py::enum_<Format>(m, "Format")
        .value("FLOAT2", Format::FLOAT2)
        .value("FLOAT3", Format::FLOAT3)
        .value("FLOAT4", Format::FLOAT4)
        .export_values();

    py::class_<MouseState>(m, "MouseState")
        .def_readonly("dx", &MouseState::dx)
        .def_readonly("dy", &MouseState::dy);

    py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer");
    
    py::class_<ShaderModule, std::shared_ptr<ShaderModule>>(m, "ShaderModule");

    py::class_<Pipeline, std::shared_ptr<Pipeline>>(m, "Pipeline");

    py::class_<PipelineBuilder, std::shared_ptr<PipelineBuilder>>(m, "PipelineBuilder")
        .def("vertexShader", &PipelineBuilder::vertexShader)
        .def("fragmentShader", &PipelineBuilder::fragmentShader)
        .def("vertexFormat", &PipelineBuilder::vertexFormat)
        .def("depthTest", &PipelineBuilder::depthTest)
        .def("pushConstant", &PipelineBuilder::pushConstant)
        .def("build", [](PipelineBuilder& builder) -> py::object {
            auto res = builder.build();
            if (res) {
                return py::cast(res.value());
            } else {
                throw std::runtime_error(res.error());
            }
        });

    py::class_<CommandBuffer, std::shared_ptr<CommandBuffer>>(m, "CommandBuffer")
        .def("begin", &CommandBuffer::begin)
        .def("beginRendering", &CommandBuffer::beginRendering, py::arg("clear_color"))
        .def("endRendering", &CommandBuffer::endRendering)
        .def("setViewport", &CommandBuffer::setViewport)
        .def("setScissor", &CommandBuffer::setScissor)
        .def("bindPipeline", &CommandBuffer::bindPipeline)
        .def("bindVertexBuffer", &CommandBuffer::bindVertexBuffer)
        .def("bindIndexBuffer", &CommandBuffer::bindIndexBuffer)
        .def("draw", &CommandBuffer::draw)
        .def("drawIndexed", &CommandBuffer::drawIndexed)
        .def("pushConstants", [](CommandBuffer& cmd, std::shared_ptr<Pipeline> pipeline, ShaderStage stage, uint32_t offset, std::string_view data) {
            cmd.pushConstants(pipeline, stage, offset, static_cast<uint32_t>(data.size()), data.data());
        });

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("init", &Engine::init)
        .def("run", &Engine::run, py::arg("app_instance") = py::none())
        .def("running", &Engine::running)
        .def("stop", &Engine::stop)
        .def("onError", &Engine::on_error)
        .def("onFrame", &Engine::on_frame)
        .def("log", &Engine::log)
        .def("getMouseState", &Engine::get_mouse_state)
        .def("isKeyPressed", &Engine::is_key_pressed)
        .def("createBuffer", &Engine::create_buffer)
        .def("createCommandBuffer", &Engine::create_command_buffer)
        .def("createPipeline", &Engine::create_pipeline)
        .def("compileShader", &Engine::compile_shader)
        .def("submit", &Engine::submit);
}