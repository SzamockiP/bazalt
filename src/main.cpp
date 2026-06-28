#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <print>
#include <atomic>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdlib>

#include "Logger.hpp"
#include "window.hpp"

extern "C" void sigint_handler(int signum)
{
    std::print("\n[Engine] SIGINT captured. Shutting down immediately...\n");
    std::exit(130);
}

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
        window_ = std::make_unique<Window>( width, height, title );
    }

    void run(py::function function)
    {

        std::signal(SIGINT, sigint_handler);

        is_running_.store(true, std::memory_order_relaxed);

        {
            py::gil_scoped_release release;

            logic_thread_ = std::jthread([this, function]()
                {
                    py::gil_scoped_acquire acquire;

                    try
                    {
                        function();
                    }
                    catch (const std::exception& e)
                    {
                        logger_.log(e.what());
                        stop();
                    }

                    stop();
                });

            while (is_running_.load(std::memory_order_relaxed))
            {
                window_->poll_events();
                if (window_->should_close())
                {
                    stop();
                }
            }

            if (logic_thread_.joinable())
            {
                logic_thread_.request_stop();
                logic_thread_.join();
            }
        }
        
        logger_.shutdown();

        std::signal(SIGINT, SIG_DFL);
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

    MouseState get_mouse_state() const
    {
        return window_->get_mouse_state();
    }

    bool is_key_pressed(int key) const
    {
        return window_->is_key_pressed(key);
    }

private:
    std::atomic<bool> is_running_{false};
    Logger logger_;
    std::jthread logic_thread_;

    std::unique_ptr<Window> window_;
};

PYBIND11_MODULE(lumapy, m){
    m.doc() = "LumaPy module";

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("init", &Engine::init)
        .def("run", &Engine::run)
        .def("running", &Engine::running)
        .def("stop", &Engine::stop)
        .def("onError", &Engine::on_error)
        .def("log", &Engine::log)
        .def("getMouseState", &Engine::get_mouse_state)
        .def("isKeyPressed", &Engine::is_key_pressed);
}