#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <print>
#include <atomic>
#include <expected>
#include <chrono>

#include "Logger.hpp"
#include "window.hpp"
#include "Renderer.hpp"
#include "Buffer.hpp"
#include "ShaderCompiler.hpp"
#include "Pipeline.hpp"
#include "CommandBuffer.hpp"
#include "Texture.hpp"

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
        delta_time_ = 0.0f;
        elapsed_time_ = 0.0f;
        frame_count_ = 0;
        is_running_.store(true, std::memory_order_relaxed);

        try
        {
            py::gil_scoped_release release;

            auto start_time = std::chrono::high_resolution_clock::now();
            auto last_frame_time = start_time;

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
                    
                    auto now = std::chrono::high_resolution_clock::now();
                    delta_time_ = std::chrono::duration<float>(now - last_frame_time).count();
                    elapsed_time_ = std::chrono::duration<float>(now - start_time).count();
                    last_frame_time = now;
                    frame_count_++;

                    renderer_->begin_frame();

                    if (renderer_->frame_skipped()) {
                        continue;
                    }
                    
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

    void set_title(const std::string& title)
    {
        if (window_) {
            window_->set_title(title);
        }
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

    bool is_mouse_button_pressed(int button) const
    {
        return window_->is_mouse_button_pressed(button);
    }

    void set_cursor_mode(int mode)
    {
        window_->set_cursor_mode(mode);
    }

    float get_delta_time() const { return delta_time_; }
    float get_time() const { return elapsed_time_; }
    uint64_t get_frame_count() const { return frame_count_; }

    int get_width() const {
        if (window_) return window_->get_width();
        return 0;
    }

    int get_height() const {
        if (window_) return window_->get_height();
        return 0;
    }

    py::object create_buffer(py::list list, BufferType type, std::optional<DataType> dataType = std::nullopt) {
        if (list.empty()) {
            throw std::runtime_error("Cannot create buffer from empty list");
        }

        size_t count = list.size();
        DataType actualType = DataType::FLOAT;

        if (dataType.has_value()) {
            actualType = dataType.value();
        } else {
            if (py::isinstance<py::float_>(list[0])) {
                actualType = DataType::FLOAT;
            } else if (py::isinstance<py::int_>(list[0])) {
                actualType = (type == BufferType::INDEX) ? DataType::UINT32 : DataType::INT32;
            } else {
                throw std::runtime_error("Could not infer data type from list elements");
            }
        }

        std::expected<std::shared_ptr<Buffer>, std::string> res = std::unexpected("Unknown data type");

        if (actualType == DataType::FLOAT) {
            std::vector<float> data(count);
            for(size_t i=0; i<count; ++i) data[i] = list[i].cast<float>();
            res = Buffer::create(*renderer_, data.data(), count * sizeof(float), type);
        } else if (actualType == DataType::UINT32) {
            std::vector<uint32_t> data(count);
            for(size_t i=0; i<count; ++i) data[i] = list[i].cast<uint32_t>();
            res = Buffer::create(*renderer_, data.data(), count * sizeof(uint32_t), type);
        } else if (actualType == DataType::UINT16) {
            std::vector<uint16_t> data(count);
            for(size_t i=0; i<count; ++i) data[i] = list[i].cast<uint16_t>();
            res = Buffer::create(*renderer_, data.data(), count * sizeof(uint16_t), type);
        } else if (actualType == DataType::INT32) {
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

    py::object create_buffer_from_numpy(py::buffer b, BufferType type) {
        py::buffer_info info = b.request();
        auto res = Buffer::create(*renderer_, info.ptr, info.size * info.itemsize, type);
        if (res) {
            return py::cast(res.value());
        } else {
            logger_.log(res.error());
            throw std::runtime_error(res.error());
        }
    }

    py::object create_empty_buffer(size_t size_in_bytes, BufferType type) {
        auto res = Buffer::create(*renderer_, nullptr, size_in_bytes, type);
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

    py::object load_texture(const std::string& path) {
        auto res = Texture::create(*renderer_, path);
        if (res) {
            return py::cast(res.value());
        } else {
            logger_.log(res.error());
            throw std::runtime_error(res.error());
        }
    }

    void submit(std::shared_ptr<CommandBuffer> cmd) {
        VkCommandBuffer vkCmd = cmd->get();
        vkResetCommandBuffer(vkCmd, 0);

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr
        };
        
        if (vkBeginCommandBuffer(vkCmd, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }

        cmd->execute(vkCmd);

        if (vkEndCommandBuffer(vkCmd) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer!");
        }

        renderer_->end_frame(vkCmd);
    }

private:
    std::atomic<bool> is_running_{false};
    Logger logger_;
    std::jthread logic_thread_;

    std::unique_ptr<Window> window_;
    std::unique_ptr<Renderer> renderer_;

    py::function frame_function_;

    float delta_time_ = 0.0f;
    float elapsed_time_ = 0.0f;
    uint64_t frame_count_ = 0;
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "LumaPy native core module";

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

    py::class_<MouseState>(m, "MouseState")
        .def_readonly("dx", &MouseState::dx)
        .def_readonly("dy", &MouseState::dy);

    py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer")
        .def("update", [](Buffer& buffer, std::string_view data) {
            buffer.update(data.data(), data.size());
        })
        .def("update", [](Buffer& buffer, py::buffer b) {
            py::buffer_info info = b.request();
            buffer.update(info.ptr, info.size * info.itemsize);
        })
        .def("update", [](Buffer& buffer, py::list list, std::optional<DataType> dataType = std::nullopt) {
            if (list.empty()) return;
            
            DataType actualType = DataType::FLOAT;
            if (dataType.has_value()) {
                actualType = dataType.value();
            } else {
                if (py::isinstance<py::float_>(list[0])) {
                    actualType = DataType::FLOAT;
                } else if (py::isinstance<py::int_>(list[0])) {
                    actualType = DataType::INT32;
                } else {
                    throw std::runtime_error("Could not infer data type from list elements");
                }
            }

            size_t count = list.size();
            if (actualType == DataType::FLOAT) {
                std::vector<float> data(count);
                for(size_t i=0; i<count; ++i) data[i] = list[i].cast<float>();
                buffer.update(data.data(), count * sizeof(float));
            } else if (actualType == DataType::UINT32) {
                std::vector<uint32_t> data(count);
                for(size_t i=0; i<count; ++i) data[i] = list[i].cast<uint32_t>();
                buffer.update(data.data(), count * sizeof(uint32_t));
            } else if (actualType == DataType::INT32) {
                std::vector<int32_t> data(count);
                for(size_t i=0; i<count; ++i) data[i] = list[i].cast<int32_t>();
                buffer.update(data.data(), count * sizeof(int32_t));
            } else {
                throw std::runtime_error("Unsupported data type for list update");
            }
        }, py::arg("list"), py::arg("dataType") = py::none());
    
    py::class_<ShaderModule, std::shared_ptr<ShaderModule>>(m, "ShaderModule");

    py::class_<Texture, std::shared_ptr<Texture>>(m, "Texture")
        .def_property_readonly("width", &Texture::width)
        .def_property_readonly("height", &Texture::height);

    py::class_<Pipeline, std::shared_ptr<Pipeline>>(m, "Pipeline");

    py::class_<PipelineBuilder, std::shared_ptr<PipelineBuilder>>(m, "PipelineBuilder")
        .def("vertexShader", &PipelineBuilder::vertexShader)
        .def("fragmentShader", &PipelineBuilder::fragmentShader)
        .def("vertexFormat", &PipelineBuilder::vertexFormat)
        .def("depthTest", &PipelineBuilder::depthTest)
        .def("cullMode", &PipelineBuilder::cullMode)
        .def("blend", &PipelineBuilder::blend)
        .def("pushConstant", &PipelineBuilder::pushConstant)
        .def("uniformBuffer", &PipelineBuilder::uniformBuffer)
        .def("storageBuffer", &PipelineBuilder::storageBuffer)
        .def("texture", &PipelineBuilder::texture)
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
        .def("drawIndexedInstanced", &CommandBuffer::drawIndexedInstanced)
        .def("pushConstants", [](CommandBuffer& cmd, std::shared_ptr<Pipeline> pipeline, ShaderStage stage, uint32_t offset, std::string_view data) {
            cmd.pushConstants(pipeline, stage, offset, static_cast<uint32_t>(data.size()), data.data());
        })
        .def("bindUniformBuffer", &CommandBuffer::bindUniformBuffer)
        .def("bindStorageBuffer", &CommandBuffer::bindStorageBuffer)
        .def("bindTexture", &CommandBuffer::bindTexture);

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("init", &Engine::init)
        .def("run", &Engine::run, py::arg("app_instance") = py::none())
        .def("running", &Engine::running)
        .def("stop", &Engine::stop)
        .def("onError", &Engine::on_error)
        .def("onFrame", &Engine::on_frame)
        .def("log", &Engine::log)
        .def("setTitle", &Engine::set_title)
        .def("getMouseState", &Engine::get_mouse_state)
        .def("isKeyPressed", &Engine::is_key_pressed)
        .def("isMouseButtonPressed", &Engine::is_mouse_button_pressed)
        .def("setCursorMode", &Engine::set_cursor_mode)
        .def("createBuffer", &Engine::create_buffer, py::arg("list"), py::arg("type"), py::arg("dataType") = py::none())
        .def("createBuffer", &Engine::create_buffer_from_numpy, py::arg("array"), py::arg("type"))
        .def("createBuffer", &Engine::create_empty_buffer, py::arg("size_in_bytes"), py::arg("type"))
        .def("createCommandBuffer", &Engine::create_command_buffer)
        .def("createPipeline", &Engine::create_pipeline)
        .def("compileShader", &Engine::compile_shader)
        .def("loadTexture", &Engine::load_texture)
        .def("submit", &Engine::submit)
        .def("getDeltaTime", &Engine::get_delta_time)
        .def("getTime", &Engine::get_time)
        .def("getFrameCount", &Engine::get_frame_count)
        .def("getWidth", &Engine::get_width)
        .def("getHeight", &Engine::get_height);

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