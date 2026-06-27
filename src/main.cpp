#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <print>
#include <atomic>
#include <thread>
#include <chrono>

namespace py = pybind11;


class Engine {
    public:
    Engine() 
    {

    }

    void init() 
    {

    }

    void run(py::function function)
    {
        _is_running.store(true, std::memory_order_relaxed);

        py::gil_scoped_release release;

        std::jthread logic_thread([this, function]()
            {
                py::gil_scoped_acquire acquire;

                try
                {
                    function();
                }
                catch (const std::exception& e)
                {
                    std::print("[Error]: {}\n", e.what());
                    stop();
                }
            });

        while (_is_running.load(std::memory_order_relaxed))
        {
            std::print("[Running]");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    bool running() const
    {
        return _is_running.load(std::memory_order_relaxed);
    }

    void stop()
    {
        _is_running.store(false, std::memory_order_relaxed);
    }

private:
    std::atomic<bool> _is_running{false};
};

PYBIND11_MODULE(lumapy, m){
    m.doc() = "LumaPy module";

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("init", &Engine::init)
        .def("run", &Engine::run)
        .def("running", &Engine::running)
        .def("stop", &Engine::stop);
}