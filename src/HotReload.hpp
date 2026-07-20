#pragma once
#include <volk.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <set>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Context.hpp"
#include "ShaderCompiler.hpp"
#include "Pipeline.hpp"
#include "Image.hpp"
#include "UploadManager.hpp"

// Watches the files bazalt itself loaded — shaders (and their #includes) and
// images — and, when one changes on disk, recompiles/re-uploads it in place.
// A shader edit rebuilds every pipeline built from it; an image edit re-uploads
// into the same VkImage. A bad edit (typo, wrong-size image, corrupt file) logs
// and keeps the last good version rendering: the whole point is that a mistake
// mid-session never takes the application down.
//
// Two threads meet here. A jthread polls file mtimes and pushes changed paths
// onto an MpscQueue — it touches no Vulkan, no Python, and never dereferences a
// watched resource. The MAIN thread drains that queue (from the frame path and
// the headless submit) and does all the real work: recompiling with the unlocked
// RecordingIncluder, and calling vkCreate* / the upload worker. Everything a
// watcher registers is held weakly, so a dropped Pipeline/Image is simply pruned
// on the next drain rather than kept alive.
class HotReloadWatcher final : public HotReloadBase
{
public:
	explicit HotReloadWatcher(Context& context)
		: context_(context), poll_interval_(std::chrono::milliseconds(poll_ms_from_env()))
	{
		thread_ = std::jthread([this](std::stop_token stop) { poll_(stop); });
	}

	// The jthread destructor requests the stop and joins; the interruptible wait
	// wakes immediately on the request.
	~HotReloadWatcher() override = default;

	HotReloadWatcher(const HotReloadWatcher&) = delete;
	HotReloadWatcher& operator=(const HotReloadWatcher&) = delete;

	void watch_shader(std::shared_ptr<ShaderModule> module) override
	{
		std::lock_guard lock(mutex_);
		ensure_watched_(module->path());
		for (const auto& inc : module->includes())
		{
			ensure_watched_(inc);
		}
		shaders_.push_back(std::move(module));
	}

	void watch_pipeline(std::shared_ptr<Pipeline> pipeline) override
	{
		std::lock_guard lock(mutex_);
		pipelines_.push_back(std::move(pipeline));
	}

	void watch_image(std::shared_ptr<Image> image, std::string path) override
	{
		std::lock_guard lock(mutex_);
		ensure_watched_(path);
		images_.emplace_back(std::move(path), std::move(image));
	}

	// Main thread only. Apply everything that changed since the last call.
	void drain() override
	{
		// Cheap in the steady state: an empty queue is a single atomic load.
		std::set<std::string> changed;
		while (std::optional<std::string> path = changed_.pop())
		{
			changed.insert(std::move(*path));
		}
		if (changed.empty())
		{
			return;
		}

		std::lock_guard lock(mutex_);

		// 1. Recompile the shaders whose file (or one of their includes) changed.
		//    A failure logs (ErrorCode::Shader -> Source::Shader, path/line already
		//    parsed) and leaves the old module in place.
		std::vector<ShaderModule*> replaced;
		for (const auto& weak : shaders_)
		{
			auto module = weak.lock();
			if (!module || !shader_affected_(*module, changed))
			{
				continue;
			}
			auto parts = ShaderCompiler::compile_parts(context_, module->path(), module->stage());
			if (!parts)
			{
				log_(parts.error());
				continue;  // old module stays; the app keeps rendering
			}
			module->replace(parts->module, std::move(parts->includes), std::move(parts->spirv));
			replaced.push_back(module.get());
		}

		// 2. Rebuild every live pipeline built from a replaced module. rebuild()
		//    swaps the VkPipeline in place on success (deferred lambdas pick it up
		//    on their next replay) and leaves the old one on failure.
		if (!replaced.empty())
		{
			for (const auto& weak : pipelines_)
			{
				auto pipeline = weak.lock();
				if (!pipeline)
				{
					continue;
				}
				const bool affected = std::ranges::any_of(
					replaced, [&](ShaderModule* m) { return pipeline->uses(m); });
				if (!affected)
				{
					continue;
				}
				if (auto r = pipeline->rebuild(); !r)
				{
					log_(r.error());
				}
			}
		}

		// 3. Re-upload the images whose file changed, through the upload worker
		//    (stbi stays on its one thread). A bad file / wrong size warns there
		//    and keeps the old contents.
		if (auto* manager = static_cast<UploadManager*>(context_.upload_manager()))
		{
			for (const auto& [path, weak] : images_)
			{
				if (!changed.contains(path))
				{
					continue;
				}
				if (auto image = weak.lock())
				{
					manager->reload(std::move(image), path);
				}
			}
		}

		// 4. Re-derive the watch set from the live registrants: freshly discovered
		//    includes start being watched, files of dropped resources fall out.
		rebuild_table_();
	}

private:
	struct WatchEntry
	{
		std::filesystem::file_time_type mtime{};
		// A change is acted on only after two consecutive polls agree on the new
		// mtime — editors save in bursts (truncate-then-write, rename-replace),
		// and a half-written read would just produce a transient ShaderError.
		std::optional<std::filesystem::file_time_type> pending;
	};

	static unsigned poll_ms_from_env()
	{
		// A test/CI knob, not part of the public API — the Python surface stays at
		// exactly one kwarg (hot_reload=True). 250 ms is imperceptible for editing.
		if (const char* raw = std::getenv("BAZALT_HOT_RELOAD_POLL_MS"))
		{
			try
			{
				const int value = std::stoi(raw);
				if (value >= 1)
				{
					return static_cast<unsigned>(value);
				}
			}
			catch (const std::exception&)
			{
				// Fall through to the default on garbage.
			}
		}
		return 250;
	}

	// caller holds mutex_. Seed a path at its current mtime so registering (or
	// re-deriving) it does not immediately look like a change.
	void ensure_watched_(const std::string& path)
	{
		if (table_.contains(path))
		{
			return;
		}
		WatchEntry entry;
		std::error_code ec;
		auto mtime = std::filesystem::last_write_time(path, ec);
		if (!ec)
		{
			entry.mtime = mtime;
		}
		table_.emplace(path, entry);
	}

	static bool shader_affected_(const ShaderModule& module, const std::set<std::string>& changed)
	{
		if (changed.contains(module.path()))
		{
			return true;
		}
		return std::ranges::any_of(module.includes(),
			[&](const std::string& inc) { return changed.contains(inc); });
	}

	// caller holds mutex_.
	void rebuild_table_()
	{
		std::erase_if(shaders_, [](const auto& w) { return w.expired(); });
		std::erase_if(pipelines_, [](const auto& w) { return w.expired(); });
		std::erase_if(images_, [](const auto& p) { return p.second.expired(); });

		std::unordered_map<std::string, WatchEntry> fresh;
		auto carry = [&](const std::string& path) {
			if (fresh.contains(path))
			{
				return;
			}
			if (auto it = table_.find(path); it != table_.end())
			{
				fresh.emplace(path, it->second);  // keep the known mtime + debounce state
			}
			else
			{
				WatchEntry entry;
				std::error_code ec;
				auto mtime = std::filesystem::last_write_time(path, ec);
				if (!ec)
				{
					entry.mtime = mtime;
				}
				fresh.emplace(path, entry);
			}
		};

		for (const auto& weak : shaders_)
		{
			if (auto module = weak.lock())
			{
				carry(module->path());
				for (const auto& inc : module->includes())
				{
					carry(inc);
				}
			}
		}
		for (const auto& [path, weak] : images_)
		{
			if (!weak.expired())
			{
				carry(path);
			}
		}
		table_ = std::move(fresh);
	}

	void log_(const Error& error)
	{
		if (auto logger = context_.logger())
		{
			logger->log(error);
		}
	}

	void poll_(std::stop_token stop)
	{
		while (!stop.stop_requested())
		{
			std::unique_lock lock(mutex_);
			// Interruptible wait: wakes on the poll interval or on stop. The lock
			// is released while waiting, so drain() and the watch_* calls run.
			cv_.wait_for(lock, stop, poll_interval_, [] { return false; });
			if (stop.stop_requested())
			{
				break;
			}
			for (auto& [path, entry] : table_)
			{
				std::error_code ec;
				auto mtime = std::filesystem::last_write_time(path, ec);
				if (ec)
				{
					continue;  // mid-rename / momentarily gone — try again next poll
				}
				// != rather than > : a copy-over can move the timestamp backwards.
				if (mtime != entry.mtime)
				{
					if (entry.pending && *entry.pending == mtime)
					{
						entry.mtime = mtime;
						entry.pending.reset();
						changed_.emplace(path);
					}
					else
					{
						entry.pending = mtime;  // first sighting — confirm next poll
					}
				}
				else
				{
					entry.pending.reset();
				}
			}
		}
	}

	Context& context_;
	std::chrono::milliseconds poll_interval_;

	std::mutex mutex_;
	std::condition_variable_any cv_;
	std::vector<std::weak_ptr<ShaderModule>> shaders_;
	std::vector<std::weak_ptr<Pipeline>> pipelines_;
	std::vector<std::pair<std::string, std::weak_ptr<Image>>> images_;
	std::unordered_map<std::string, WatchEntry> table_;

	MpscQueue<std::string> changed_;

	// Last member: the jthread joins before anything it touches is destroyed.
	std::jthread thread_;
};
