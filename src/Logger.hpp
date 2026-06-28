#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <semaphore>
#include <thread>

#include "MpscQueue.hpp"

namespace py = pybind11;

class Logger
{
public:
	Logger()
	{
		log_thread_ = std::jthread([this](std::stop_token stop_token)
			{
				while (true)
				{
					error_semaphore_.acquire();

					if (stop_token.stop_requested())
						break;

					drain_messages();
				}

				drain_messages();
			});
	}

	~Logger()
	{
		shutdown();
	}

	void shutdown()
	{
		if (!log_thread_.joinable())
			return;

		log_thread_.request_stop();
		error_semaphore_.release(); 

		py::gil_scoped_release release;
		log_thread_.join();
	}

	void register_callback(py::function callback)
	{
		std::lock_guard<std::mutex> lock(_callbacks_mutex);
		error_callbacks_.push_back(callback);
	}
	
	void log(const std::string& message)
	{
		error_messages_.push(message);
		error_semaphore_.release();
	}
private:

	void drain_messages()
	{
		py::gil_scoped_acquire gil;
		std::lock_guard<std::mutex> lock(_callbacks_mutex);
		while (std::optional<std::string> msg = error_messages_.pop())
		{
			for (const auto& callback : error_callbacks_)
			{
				callback(*msg);
			}
		}
	}

	MpscQueue<std::string> error_messages_{};
	std::counting_semaphore<> error_semaphore_{ 0 };

	std::mutex _callbacks_mutex;
	std::vector<py::function> error_callbacks_{};

	std::jthread log_thread_;
};
