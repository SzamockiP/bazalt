#pragma once
#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <atomic>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "Error.hpp"
#include "Features.hpp"
#include "Logger.hpp"

// How hard to try to turn on the validation layers.
//
// Auto is the default and never fails: end-user machines generally have no
// layers installed, and a missing layer is not a reason to refuse to render.
enum class ValidationMode
{
	Off,
	Auto,
	On,
};

// Everything the caller can ask of a Context, in capability terms.
struct ContextConfig
{
	ValidationMode validation = ValidationMode::Auto;
	std::vector<Feature> required;
	std::vector<Feature> optional;

	// How many frames may be recorded ahead of the GPU. 2 is the classic
	// latency/throughput trade-off; 1 is legal and useful for debugging.
	std::uint32_t frames_in_flight = 2;

	// Escape hatch, documented as "you shouldn't need this". Present so that the
	// capability abstraction never becomes a ceiling.
	std::vector<std::string> raw_extensions;
};

class Context : public std::enable_shared_from_this<Context>
{
public:
	static std::expected<std::shared_ptr<Context>, Error> create(
		std::shared_ptr<Logger> logger,
		const ContextConfig& config = {})
	{
		// volk installs its function pointers as globals, and volkLoadDevice binds
		// them to one specific VkDevice. A second live Context therefore silently
		// redirects the first one's GPU calls at its own device — an access
		// violation with no diagnostic whatsoever. Sequential create/destroy is
		// fine; only overlap is not.
		//
		// The real fix is volkLoadDeviceTable + per-Context dispatch tables, which
		// touches every vk* call site. Until that's worth doing, fail loudly.
		if (live_contexts_.load() > 0)
		{
			return std::unexpected(err_init(
				"Only one bazalt Context can exist at a time. volk binds its global "
				"function pointers to a single device, so a second Context would "
				"silently corrupt the first. Destroy the existing Context before "
				"creating another (creating them one after another is fine)."));
		}

		if (config.frames_in_flight < 1 || config.frames_in_flight > 4)
		{
			return std::unexpected(err_init(std::format(
				"frames_in_flight must be between 1 and 4, got {}", config.frames_in_flight)));
		}

		auto context = std::shared_ptr<Context>(new Context(logger));
		context->frames_in_flight_ = config.frames_in_flight;

		auto target_api = create_instance_(*context, config, logger);
		if (!target_api)
		{
			return std::unexpected(target_api.error());
		}
		if (auto r = select_physical_device_(*context, config, *target_api); !r)
		{
			return std::unexpected(r.error());
		}
		if (auto r = configure_features_(*context, config, logger, *target_api); !r)
		{
			return std::unexpected(r.error());
		}
		if (auto r = create_device_(*context); !r)
		{
			return std::unexpected(r.error());
		}
		if (auto r = create_allocator_and_pool_(*context); !r)
		{
			return std::unexpected(r.error());
		}

		if (logger)
		{
			// Spell out which path was negotiated: a bug report from an unfamiliar
			// machine is unreadable without knowing whether it took the 1.3-core or
			// the 1.2+KHR route, and whether it went headless.
			logger->log(Severity::Info, Source::General,
				std::format("Vulkan: Initialized ({}, API {}, dynamic rendering: {}{})",
					context->device_name(),
					api_version_string(context->api_version()),
					context->dynamic_rendering_khr_ ? "KHR extension" : "core",
					context->headless_ ? ", headless" : ""));
		}

		// Last thing before handing the Context over: only a fully built Context
		// counts as live, so a failed create() doesn't leave the slot occupied.
		context->registered_ = true;
		live_contexts_.fetch_add(1);

		return context;
	}

	~Context()
	{
		if (registered_)
		{
			live_contexts_.fetch_sub(1);
		}

		if (vkb_device_.device)
		{
			vkDeviceWaitIdle(vkb_device_.device);
		}

		if (command_pool_)
		{
			vkDestroyCommandPool(vkb_device_.device, command_pool_, nullptr);
		}

		if (allocator_)
		{
			vmaDestroyAllocator(allocator_);
		}

		vkb::destroy_device(vkb_device_);
		vkb::destroy_instance(vkb_instance_);
	}

	Context(const Context&) = delete;
	Context& operator=(const Context&) = delete;

	VkInstance instance() const { return vkb_instance_.instance; }
	VkDevice device() const { return vkb_device_.device; }
	VkPhysicalDevice physical_device() const { return vkb_physical_device_.physical_device; }
	VkQueue graphics_queue() const { return graphics_queue_; }
	std::uint32_t graphics_queue_family() const { return graphics_queue_family_; }
	VmaAllocator allocator() const { return allocator_; }
	VkCommandPool command_pool() const { return command_pool_; }
	std::shared_ptr<Logger> logger() const { return logger_; }

	// Internal — for use by renderers and other subsystems
	const vkb::Instance& vkb_instance() const { return vkb_instance_; }
	const vkb::Device& vkb_device() const { return vkb_device_; }

	// ── Capabilities ──────────────────────────────────────────────────────────

	bool supports(Feature feature) const { return enabled_features_.contains(feature); }

	std::string device_name() const
	{
		return vkb_physical_device_.properties.deviceName;
	}

	// The *negotiated* version — what the device was actually created against —
	// not the raw properties.apiVersion. A 1.3-capable GPU behind a 1.2 loader
	// runs the 1.2+KHR path, and shaders compiled for 1.3 would be invalid
	// there. Context stays ignorant of shaderc: ShaderCompiler maps this onto
	// a shaderc_env_version itself.
	std::uint32_t api_version() const
	{
		return negotiated_api_version_;
	}

	// True when no windowing extensions were available, so no SwapchainRenderer
	// can be created against this Context.
	bool headless() const { return headless_; }
	bool swapchain_supported() const { return swapchain_supported_; }

	// ── Frame ring ────────────────────────────────────────────────────────────
	//
	// The Context owns and advances the frame counter; renderers and the
	// headless submit path both call begin_frame_internal() when a new frame
	// starts. A monotonic serial rather than a wrapping index, because the
	// deletion queue and upload bookkeeping need "how far has the GPU
	// progressed", which a modulo index cannot answer.
	std::uint32_t frames_in_flight() const { return frames_in_flight_; }
	std::uint64_t frame_serial() const { return frame_serial_; }
	std::uint32_t frame_index() const
	{
		return static_cast<std::uint32_t>(frame_serial_ % frames_in_flight_);
	}

	// Not exposed to Python: headless users have exactly one verb (ctx.submit),
	// and a second frame-advancing verb would be a footgun until multiple
	// submits per frame exist.
	//
	// Call sites pick the boundary that keeps `buffer.update()` and the submit
	// that consumes it on the SAME ring slot: SwapchainRenderer::begin_frame
	// advances on entry (updates happen between begin and submit), the headless
	// ctx.submit advances after submitting (updates happen before the call).
	std::uint64_t advance_frame()
	{
		return ++frame_serial_;
	}

	// Guards against two SwapchainRenderers fighting over the frame ring.
	bool has_swapchain_renderer() const { return has_swapchain_renderer_; }
	void set_has_swapchain_renderer(bool value) { has_swapchain_renderer_ = value; }

	// VkQueue is externally synchronized. Today every submit happens on the main
	// thread, so this mutex is uncontended — it exists because 0.5's upload
	// worker submits from its own thread, and every vkQueueSubmit/Present/
	// WaitIdle must hold it from then on.
	std::mutex& queue_mutex() { return queue_mutex_; }

private:
	Context(std::shared_ptr<Logger> logger) : logger_(logger) {}

	// ── create() steps ────────────────────────────────────────────────────────

	// volk + instance (with validation and the headless fallback). Returns the
	// negotiated instance API version.
	static std::expected<std::uint32_t, Error> create_instance_(
		Context& ctx, const ContextConfig& config, const std::shared_ptr<Logger>& logger)
	{
		if (auto e = check(volkInitialize(), "initialize volk"))
		{
			return std::unexpected(*e);
		}

		auto system_info = vkb::SystemInfo::get_system_info();
		if (!system_info)
		{
			return std::unexpected(err_init("Vulkan: " + system_info.error().message()));
		}

		// Take 1.3 where it exists, otherwise 1.2. Requiring 1.3 outright — as this
		// used to — rejects older Intel iGPUs, MoltenVK and any driver still on 1.2,
		// which is a large slice of real machines. Everything bazalt needs from 1.3
		// is available on 1.2 through VK_KHR_dynamic_rendering.
		const bool has_1_3 = system_info->is_instance_version_available(1, 3);
		const std::uint32_t target_api = has_1_3 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_2;

		// Instance + Debug Messenger
		auto inst_builder = vkb::InstanceBuilder{}
			.set_app_name("Bazalt Engine")
			.set_app_version(1, 0, 0)
			.require_api_version(target_api);

		// Validation is independent of whether a logger was supplied. It used to
		// be gated on `logger != nullptr`, which meant the default path rendered
		// with the layers off and stayed silent about its own bugs.
		if (config.validation != ValidationMode::Off)
		{
			if (config.validation == ValidationMode::On)
			{
				inst_builder.enable_validation_layers();
			}
			else
			{
				// Enables the layers only if they are actually present.
				inst_builder.request_validation_layers();
			}

			inst_builder
				.set_debug_callback(debug_callback)
				.set_debug_callback_user_data_pointer(logger.get())
				.set_debug_messenger_severity(
					VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
					VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
				.set_debug_messenger_type(
					VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
					VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
					VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
		}

		// Surface extensions are NOT enabled by hand. vk-bootstrap already adds the
		// right ones per platform (win32/xcb/xlib/wayland/metal), and the previous
		// hand-rolled enable_extension("VK_KHR_surface") was both redundant and a
		// hard failure path: enable_extension refuses to build when absent.
		// It also silently ignored the extension list GLFW reports.
		for (const auto& extension : config.raw_extensions)
		{
			inst_builder.enable_extension(extension.c_str());
		}

		auto inst_ret = inst_builder.build();

		// A machine with no display has no windowing extensions, and vk-bootstrap
		// treats that as fatal. Fall back to a headless instance instead: whether a
		// window is involved is decided later by creating a SwapchainRenderer (or
		// not), so Context has no business demanding one — and no business making
		// the user pass headless=True to say so.
		if (!inst_ret && inst_ret.error() == vkb::InstanceError::windowing_extensions_not_present)
		{
			inst_builder.set_headless(true);
			inst_ret = inst_builder.build();
			if (inst_ret)
			{
				ctx.headless_ = true;
				if (logger)
				{
					logger->log(Severity::Info, Source::General,
						"Vulkan: no windowing extensions present, continuing headless");
				}
			}
		}

		if (!inst_ret)
		{
			Error error = err_init("Vulkan: " + inst_ret.error().message());
			error.result = inst_ret.vk_result();
			return std::unexpected(error);
		}
		ctx.vkb_instance_ = inst_ret.value();
		volkLoadInstance(ctx.vkb_instance_.instance);

		return target_api;
	}

	static std::expected<void, Error> select_physical_device_(
		Context& ctx, const ContextConfig& config, std::uint32_t target_api)
	{
		// Nothing is required here beyond the API version: swapchain support used to
		// be a required extension, which rejected headless-only GPUs outright and
		// made the require_present(false) on the next line pointless. Optional bits
		// are enabled per-device in configure_features_, once we know what this
		// device actually has.
		auto selector = vkb::PhysicalDeviceSelector{ ctx.vkb_instance_ }
			.set_minimum_version(VK_API_VERSION_MAJOR(target_api), VK_API_VERSION_MINOR(target_api))
			.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
			.require_present(false);

		// Required features must gate *selection*, so a machine with two GPUs picks
		// the one that can do the job rather than failing on the preferred one.
		VkPhysicalDeviceFeatures required_features{};
		for (Feature feature : config.required)
		{
			enable_feature(required_features, feature);
		}
		selector.set_required_features(required_features);

		auto phys_ret = selector.select();
		if (!phys_ret)
		{
			std::string detail;
			for (Feature feature : config.required)
			{
				detail += (detail.empty() ? "" : ", ");
				detail += feature_name(feature);
			}
			Error error = err_init("Vulkan: no suitable GPU found" +
				(detail.empty() ? std::string() : " for required features [" + detail + "]") +
				": " + phys_ret.error().message());
			error.result = phys_ret.vk_result();
			return std::unexpected(error);
		}
		ctx.vkb_physical_device_ = phys_ret.value();
		return {};
	}

	// EVERY enable_* call in here must happen BEFORE the DeviceBuilder exists
	// (i.e. before create_device_): its constructor takes the PhysicalDevice
	// *by value*, so anything enabled afterwards is written to a copy and
	// silently dropped. That mistake cost a swapchain that was never enabled on
	// the device — an access violation with no diagnostic.
	static std::expected<void, Error> configure_features_(
		Context& ctx, const ContextConfig& config,
		const std::shared_ptr<Logger>& logger, std::uint32_t target_api)
	{
		// vkb::PhysicalDevice::features is documented as the *selected* features,
		// not the available ones, so ask the driver directly.
		VkPhysicalDeviceFeatures available{};
		vkGetPhysicalDeviceFeatures(ctx.vkb_physical_device_.physical_device, &available);

		// Dynamic rendering: core in 1.3, an extension on 1.2. Same capability,
		// two spellings — resolve it here so nothing else has to care.
		ctx.dynamic_rendering_khr_ = false;
		const bool has_1_3 = target_api >= VK_API_VERSION_1_3;
		const bool device_has_1_3 =
			ctx.vkb_physical_device_.properties.apiVersion >= VK_API_VERSION_1_3;

		if (has_1_3 && device_has_1_3)
		{
			// core path — VkPhysicalDeviceVulkan13Features in create_device_
			ctx.negotiated_api_version_ = VK_API_VERSION_1_3;
		}
		else if (ctx.vkb_physical_device_.enable_extension_if_present(
			VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
		{
			ctx.dynamic_rendering_khr_ = true;
			ctx.negotiated_api_version_ = VK_API_VERSION_1_2;
		}
		else
		{
			return std::unexpected(err_init(
				"Vulkan: this GPU/driver supports neither Vulkan 1.3 nor "
				"VK_KHR_dynamic_rendering, which bazalt requires. Updating the "
				"graphics driver usually fixes this."));
		}

		// Presentation is optional: a headless or compute-only Context is legitimate.
		// SwapchainRenderer verifies present support at its own creation time.
		ctx.swapchain_supported_ =
			ctx.vkb_physical_device_.enable_extension_if_present(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

		// Optional features: enable what this device happens to have, and record
		// what stuck so supports() can answer honestly.
		VkPhysicalDeviceFeatures enabled_features{};
		for (Feature feature : config.required)
		{
			enable_feature(enabled_features, feature);
			ctx.enabled_features_.insert(feature);
		}
		for (Feature feature : config.optional)
		{
			if (feature_available(available, feature))
			{
				enable_feature(enabled_features, feature);
				ctx.enabled_features_.insert(feature);
			}
			else if (logger)
			{
				logger->log(Severity::Info, Source::General,
					std::format("Vulkan: optional feature {} is not supported by this GPU",
						feature_name(feature)));
			}
		}

		// Anisotropic filtering is on by default when present, because Texture uses
		// it. It used to be *required*, which turned a nicety into a reach blocker.
		if (feature_available(available, Feature::ANISOTROPIC_FILTERING))
		{
			enable_feature(enabled_features, Feature::ANISOTROPIC_FILTERING);
			ctx.enabled_features_.insert(Feature::ANISOTROPIC_FILTERING);
		}

		// Timeline semaphores pace the deletion queue and async uploads. They are
		// core in 1.2 (our floor), so the entry points are always loaded — but the
		// feature bit still has to be enabled at device creation, and checking it
		// here turns a cryptic device-creation error code into a sentence.
		VkPhysicalDeviceVulkan12Features available12{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext = nullptr
		};
		VkPhysicalDeviceFeatures2 available2{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext = &available12
		};
		vkGetPhysicalDeviceFeatures2(ctx.vkb_physical_device_.physical_device, &available2);
		if (!available12.timelineSemaphore)
		{
			return std::unexpected(err_init(
				"Vulkan: this GPU/driver does not support timeline semaphores, which "
				"bazalt requires (they are mandatory in conformant Vulkan 1.2 drivers). "
				"Updating the graphics driver usually fixes this."));
		}

		ctx.vkb_physical_device_.enable_features_if_present(enabled_features);
		return {};
	}

	// Only entered once the PhysicalDevice is final and safe to copy.
	static std::expected<void, Error> create_device_(Context& ctx)
	{
		// shaderDemoteToHelperInvocation / shaderTerminateInvocation are mandatory
		// in Vulkan 1.3, and glslang compiles `discard` to one of those opcodes
		// when targeting SPIR-V 1.6 — so a fragment shader with `discard` breaks
		// unless they are enabled alongside the 1.3 target.
		VkPhysicalDeviceVulkan13Features features13{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
			.pNext = nullptr,
			.shaderDemoteToHelperInvocation = VK_TRUE,
			.shaderTerminateInvocation = VK_TRUE,
			.dynamicRendering = VK_TRUE
		};
		VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_khr{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
			.pNext = nullptr,
			.dynamicRendering = VK_TRUE
		};

		// Core in 1.2, so this rides along on both paths. No KHR aliasing needed,
		// unlike dynamic rendering: the floor is 1.2, so the core entry points
		// (vkWaitSemaphores etc.) are always loaded.
		VkPhysicalDeviceVulkan12Features features12{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext = nullptr,
			.timelineSemaphore = VK_TRUE
		};

		auto dev_builder = vkb::DeviceBuilder{ ctx.vkb_physical_device_ };
		dev_builder.add_pNext(&features12);
		if (ctx.dynamic_rendering_khr_)
		{
			dev_builder.add_pNext(&dynamic_rendering_khr);
		}
		else
		{
			dev_builder.add_pNext(&features13);
		}

		auto dev_ret = dev_builder.build();

		if (!dev_ret)
		{
			Error error = err_init("Vulkan: " + dev_ret.error().message());
			error.result = dev_ret.vk_result();
			return std::unexpected(error);
		}
		ctx.vkb_device_ = dev_ret.value();
		volkLoadDevice(ctx.vkb_device_.device);
		ctx.alias_dynamic_rendering_entry_points();

		auto gq = ctx.vkb_device_.get_queue(vkb::QueueType::graphics);
		if (!gq)
		{
			return std::unexpected(err_init("Vulkan: Failed to get graphics queue"));
		}
		ctx.graphics_queue_ = gq.value();
		ctx.graphics_queue_family_ =
			ctx.vkb_device_.get_queue_index(vkb::QueueType::graphics).value();

		return {};
	}

	static std::expected<void, Error> create_allocator_and_pool_(Context& ctx)
	{
		VmaVulkanFunctions vulkanFunctions = {};
		vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
		vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.physicalDevice = ctx.vkb_physical_device_.physical_device;
		allocatorInfo.device = ctx.vkb_device_.device;
		allocatorInfo.instance = ctx.vkb_instance_.instance;
		allocatorInfo.pVulkanFunctions = &vulkanFunctions;
		// The negotiated version, not a hardcoded one: telling VMA 1.3 on the
		// 1.2+KHR path would let it call entry points the device never promised.
		allocatorInfo.vulkanApiVersion = ctx.negotiated_api_version_;

		if (auto e = check(vmaCreateAllocator(&allocatorInfo, &ctx.allocator_),
		                   "create VMA allocator"))
		{
			return std::unexpected(*e);
		}

		VkCommandPoolCreateInfo poolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = ctx.graphics_queue_family_
		};

		if (auto e = check(vkCreateCommandPool(ctx.vkb_device_.device, &poolInfo,
		                                       nullptr, &ctx.command_pool_),
		                   "create command pool"))
		{
			return std::unexpected(*e);
		}

		return {};
	}

	// On a 1.2 device the dynamic-rendering entry points are loaded under their
	// KHR names and the core symbols stay null. Call sites use the core names
	// (vkCmdBeginRendering), so point them at the KHR implementations here. The
	// structs are already aliases, so nothing else has to know which path we took.
	void alias_dynamic_rendering_entry_points()
	{
#if defined(VK_VERSION_1_3) && defined(VK_KHR_dynamic_rendering)
		if (!vkCmdBeginRendering && vkCmdBeginRenderingKHR)
		{
			vkCmdBeginRendering = vkCmdBeginRenderingKHR;
		}
		if (!vkCmdEndRendering && vkCmdEndRenderingKHR)
		{
			vkCmdEndRendering = vkCmdEndRenderingKHR;
		}
#endif
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
		VkDebugUtilsMessageTypeFlagsEXT message_type,
		const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
		void* user_data)
	{
		Logger* logger = static_cast<Logger*>(user_data);
		if (logger)
		{
			// The severity travels as data. It used to be glued onto the front of
			// the text as "ERROR: ", which is why the callback named on_error was
			// receiving warnings and info messages indistinguishably.
			Severity severity = Severity::Info;
			if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
			{
				severity = Severity::Error;
			}
			else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
			{
				severity = Severity::Warning;
			}

			logger->log(severity, Source::Validation, callback_data->pMessage);
		}
		return VK_FALSE;
	}

	std::shared_ptr<Logger> logger_;

	vkb::Instance vkb_instance_;
	vkb::PhysicalDevice vkb_physical_device_;
	vkb::Device vkb_device_;

	VkQueue graphics_queue_ = VK_NULL_HANDLE;
	std::uint32_t graphics_queue_family_ = 0;
	std::mutex queue_mutex_;

	VmaAllocator allocator_ = VK_NULL_HANDLE;
	VkCommandPool command_pool_ = VK_NULL_HANDLE;

	static inline std::atomic<int> live_contexts_{ 0 };

	std::set<Feature> enabled_features_;
	bool registered_ = false;
	bool has_swapchain_renderer_ = false;
	bool headless_ = false;
	bool swapchain_supported_ = false;
	bool dynamic_rendering_khr_ = false;

	// Set by configure_features_: 1.3 on the core path, 1.2 on the KHR path.
	std::uint32_t negotiated_api_version_ = VK_API_VERSION_1_2;

	std::uint32_t frames_in_flight_ = 2;
	std::uint64_t frame_serial_ = 0;
};
