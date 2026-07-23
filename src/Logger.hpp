#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "MpscQueue.hpp"

namespace py = pybind11;

// Severity is data, not a prefix baked into the message string.
//
// NOTE: the enumerators are deliberately NOT spelled ERROR/WARNING/INFO —
// <windows.h> (pulled in by volk via VK_USE_PLATFORM_WIN32_KHR) defines ERROR
// as a macro. They are exported to Python under the uppercase names in main.cpp.
enum class Severity
{
    Info = 0,
    Warning = 1,
    Error = 2,
};

// Where a message came from. Lets a callback route by subsystem instead of
// pattern-matching on the text.
enum class Source
{
    General,
    Validation,
    Window,
    Shader,
    Upload,
    Device,
};

struct LogMessage
{
    Severity severity = Severity::Info;
    Source source = Source::General;
    std::string text;
};

inline Source source_from_error_code(ErrorCode code)
{
    switch (code)
    {
        case ErrorCode::Shader:
            return Source::Shader;
        case ErrorCode::Window:
            return Source::Window;
        case ErrorCode::DeviceLost:
            return Source::Device;
        default:
            return Source::General;
    }
}

class Logger
{
public:
    explicit Logger(Severity min_severity = Severity::Warning)
        : min_severity_(min_severity)
    {
        {
            std::lock_guard lock(registry_mutex());
            registry().push_back(this);
        }

        log_thread_ = std::jthread(
            [this](std::stop_token stop_token)
            {
                while (true)
                {
                    message_semaphore_.acquire();

                    if (stop_token.stop_requested())
                        break;

                    drain_messages();
                }

                drain_messages();
            });
    }

    ~Logger()
    {
        {
            std::lock_guard lock(registry_mutex());
            std::erase(registry(), this);
        }
        shutdown();
    }

    // The drain thread calls into Python, so it must be joined BEFORE the
    // interpreter starts finalizing — a logger still alive at process exit
    // (any script that ends with its Context in scope) would otherwise crash
    // with "could not acquire lock for stderr at interpreter shutdown". The
    // module registers this with atexit, which runs while Python is intact.
    static void shutdown_all()
    {
        std::lock_guard lock(registry_mutex());
        for (Logger* logger : registry())
        {
            logger->shutdown();
        }
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void shutdown()
    {
        if (!log_thread_.joinable())
            return;

        log_thread_.request_stop();
        message_semaphore_.release();

        py::gil_scoped_release release;
        log_thread_.join();
    }

    void register_callback(py::function callback)
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        callbacks_.push_back(std::move(callback));
    }

private:
    // Every live logger, so atexit can join every drain thread. Raw pointers:
    // the destructor unregisters under the same mutex.
    static std::mutex& registry_mutex()
    {
        static std::mutex mutex;
        return mutex;
    }
    static std::vector<Logger*>& registry()
    {
        static std::vector<Logger*> loggers;
        return loggers;
    }

public:
    Severity min_severity() const
    {
        return min_severity_;
    }
    void set_min_severity(Severity severity)
    {
        min_severity_ = severity;
    }

    void log(Severity severity, Source source, std::string text)
    {
        // Filter before queueing — a suppressed message costs nothing.
        if (severity < min_severity_.load())
            return;

        pending_.fetch_add(1);
        messages_.push(LogMessage{severity, source, std::move(text)});
        message_semaphore_.release();
    }

    // Blocks until every queued message has reached its callbacks.
    //
    // Delivery is asynchronous, so without this a test asserting "no validation
    // errors occurred" is really asserting "none had arrived yet" — a sleep() with
    // extra steps. The GIL must be released or this deadlocks against the drain
    // thread, which needs it to call back into Python.
    void flush()
    {
        py::gil_scoped_release release;
        while (pending_.load() > 0)
        {
            std::this_thread::yield();
        }
    }

    // An Error is just a message that also happens to be returned. Logging it
    // keeps the severity/source mapping in one place.
    void log(const Error& error)
    {
        log(Severity::Error, source_from_error_code(error.code), error.message);
    }

private:
    void drain_messages()
    {
        py::gil_scoped_acquire gil;
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        while (std::optional<LogMessage> msg = messages_.pop())
        {
            for (const auto& callback : callbacks_)
            {
                callback(*msg);
            }
            // Decremented only after the callbacks have run, so flush() really does
            // mean "delivered", not "dequeued".
            pending_.fetch_sub(1);
        }
    }

    MpscQueue<LogMessage> messages_{};
    std::counting_semaphore<> message_semaphore_{0};
    std::atomic<int> pending_{0};

    std::atomic<Severity> min_severity_;

    std::mutex callbacks_mutex_;
    std::vector<py::function> callbacks_{};

    std::jthread log_thread_;
};
