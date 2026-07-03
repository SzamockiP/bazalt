#pragma once
#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <expected>
#include <memory>
#include <vector>
#include <string>

#include "Logger.hpp"
#include "window.hpp"

class Renderer
{
private:
	Renderer(Logger& logger, Window& window)
		: logger_(logger), window_(window) {}

public:
	static std::expected<std::unique_ptr<Renderer>, std::string> create(Logger& logger, Window& window)
	{
		if (volkInitialize() != VK_SUCCESS)
		{
			return std::unexpected("Vulkan: Failed to initialize volk");
		}

		auto renderer = std::unique_ptr<Renderer>(new Renderer(logger, window));

		// Instance + Debug Messenger 
		auto inst_ret = vkb::InstanceBuilder{}
			.set_app_name("LumaPy Engine")
			.set_app_version(1, 0, 0)
			.require_api_version(1, 3, 0)
			.request_validation_layers()
			.set_debug_callback(debug_callback)
			.set_debug_callback_user_data_pointer(&logger)
			.set_debug_messenger_severity(
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
			.set_debug_messenger_type(
				VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
			.build();

		if (!inst_ret)
		{
			return std::unexpected("Vulkan: " + inst_ret.error().message());
		}
		renderer->vkb_instance_ = inst_ret.value();
		volkLoadInstance(renderer->vkb_instance_.instance);

		// Surface 
		VkSurfaceKHR surface = VK_NULL_HANDLE;
		if (glfwCreateWindowSurface(renderer->vkb_instance_.instance,
				window.get_native_handle(), nullptr, &surface) != VK_SUCCESS)
		{
			return std::unexpected("Vulkan: Failed to create window surface");
		}
		renderer->surface_ = surface;

		// Physical Device
		auto phys_ret = vkb::PhysicalDeviceSelector{ renderer->vkb_instance_ }
			.set_surface(surface)
			.set_minimum_version(1, 3)
			.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
			.select();

		if (!phys_ret)
		{
			return std::unexpected("Vulkan: " + phys_ret.error().message());
		}
		renderer->vkb_physical_device_ = phys_ret.value();

		// Logical Device + Queues
		VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_feature{};
		dynamic_rendering_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
		dynamic_rendering_feature.dynamicRendering = VK_TRUE;

		auto dev_ret = vkb::DeviceBuilder{ renderer->vkb_physical_device_ }
			.add_pNext(&dynamic_rendering_feature)
			.build();

		if (!dev_ret)
		{
			return std::unexpected("Vulkan: " + dev_ret.error().message());
		}
		renderer->vkb_device_ = dev_ret.value();
		volkLoadDevice(renderer->vkb_device_.device);

		auto gq = renderer->vkb_device_.get_queue(vkb::QueueType::graphics);
		if (!gq)
		{
			return std::unexpected("Vulkan: Failed to get graphics queue");
		}
		renderer->graphics_queue_ = gq.value();
		renderer->graphics_queue_family_ =
			renderer->vkb_device_.get_queue_index(vkb::QueueType::graphics).value();

		auto pq = renderer->vkb_device_.get_queue(vkb::QueueType::present);
		if (!pq)
		{
			return std::unexpected("Vulkan: Failed to get present queue");
		}
		renderer->present_queue_ = pq.value();

		// Swapchain
		auto sc_ret = vkb::SwapchainBuilder{ renderer->vkb_device_ }
			.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
			.add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR)
			.build();

		if (!sc_ret)
		{
			return std::unexpected("Vulkan: " + sc_ret.error().message());
		}
		renderer->vkb_swapchain_ = sc_ret.value();
		renderer->swapchain_images_ = renderer->vkb_swapchain_.get_images().value();
		renderer->swapchain_image_views_ = renderer->vkb_swapchain_.get_image_views().value();

		// VMA Allocator
		VmaVulkanFunctions vulkanFunctions = {};
		vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
		vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

		VmaAllocatorCreateInfo allocatorInfo = {};
		allocatorInfo.physicalDevice = renderer->vkb_physical_device_.physical_device;
		allocatorInfo.device = renderer->vkb_device_.device;
		allocatorInfo.instance = renderer->vkb_instance_.instance;
		allocatorInfo.pVulkanFunctions = &vulkanFunctions;
		allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

		if (vmaCreateAllocator(&allocatorInfo, &renderer->allocator_) != VK_SUCCESS)
		{
			return std::unexpected("Vulkan: Failed to create VMA allocator");
		}

		// Command Pool
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = renderer->graphics_queue_family_;

		if (vkCreateCommandPool(renderer->vkb_device_.device, &poolInfo, nullptr, &renderer->command_pool_) != VK_SUCCESS)
		{
			return std::unexpected("Vulkan: Failed to create command pool");
		}

		// Sync Objects
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			if (vkCreateSemaphore(renderer->vkb_device_.device, &semaphoreInfo, nullptr, &renderer->image_available_semaphores_[i]) != VK_SUCCESS ||
				vkCreateFence(renderer->vkb_device_.device, &fenceInfo, nullptr, &renderer->in_flight_fences_[i]) != VK_SUCCESS)
			{
				return std::unexpected("Vulkan: Failed to create CPU synchronization objects");
			}
		}

		renderer->render_finished_semaphores_.resize(renderer->swapchain_images_.size());
		for (size_t i = 0; i < renderer->swapchain_images_.size(); i++)
		{
			if (vkCreateSemaphore(renderer->vkb_device_.device, &semaphoreInfo, nullptr, &renderer->render_finished_semaphores_[i]) != VK_SUCCESS)
			{
				return std::unexpected("Vulkan: Failed to create render finished semaphores");
			}
		}

		logger.log("Vulkan: Initialized (" +
			std::string(renderer->vkb_physical_device_.properties.deviceName) + ")");

		return renderer;
	}

	static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

	~Renderer()
	{
		if (vkb_device_.device)
		{
			vkDeviceWaitIdle(vkb_device_.device);
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			if (image_available_semaphores_[i]) vkDestroySemaphore(vkb_device_.device, image_available_semaphores_[i], nullptr);
			if (in_flight_fences_[i]) vkDestroyFence(vkb_device_.device, in_flight_fences_[i], nullptr);
		}
		
		for (auto sem : render_finished_semaphores_)
		{
			if (sem) vkDestroySemaphore(vkb_device_.device, sem, nullptr);
		}

		if (command_pool_)
		{
			vkDestroyCommandPool(vkb_device_.device, command_pool_, nullptr);
		}

		if (allocator_)
		{
			vmaDestroyAllocator(allocator_);
		}

		for (auto iv : swapchain_image_views_)
		{
			vkDestroyImageView(vkb_device_.device, iv, nullptr);
		}

		vkb::destroy_swapchain(vkb_swapchain_);

		if (surface_)
		{
			vkDestroySurfaceKHR(vkb_instance_.instance, surface_, nullptr);
		}

		vkb::destroy_device(vkb_device_);
		vkb::destroy_instance(vkb_instance_);
	}

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

	VkInstance instance() const { return vkb_instance_.instance; }
	VkDevice device() const { return vkb_device_.device; }
	VkPhysicalDevice physical_device() const { return vkb_physical_device_.physical_device; }
	VkQueue graphics_queue() const { return graphics_queue_; }
	VkQueue present_queue() const { return present_queue_; }
	std::uint32_t graphics_queue_family() const { return graphics_queue_family_; }
	VkSwapchainKHR swapchain() const { return vkb_swapchain_.swapchain; }
	VkFormat swapchain_format() const { return vkb_swapchain_.image_format; }
	VkExtent2D swapchain_extent() const { return vkb_swapchain_.extent; }
	const std::vector<VkImage>& swapchain_images() const { return swapchain_images_; }
	const std::vector<VkImageView>& swapchain_image_views() const { return swapchain_image_views_; }
	VmaAllocator allocator() const { return allocator_; }
	VkCommandPool command_pool() const { return command_pool_; }

	std::uint32_t current_frame() const { return current_frame_; }
	std::uint32_t current_image_index() const { return image_index_; }

	void begin_frame()
	{
		vkWaitForFences(vkb_device_.device, 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);

		VkResult result = vkAcquireNextImageKHR(
			vkb_device_.device,
			vkb_swapchain_.swapchain,
			UINT64_MAX,
			image_available_semaphores_[current_frame_],
			VK_NULL_HANDLE,
			&image_index_
		);

		if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			logger_.log("Failed to acquire swapchain image!");
		}

		vkResetFences(vkb_device_.device, 1, &in_flight_fences_[current_frame_]);
	}

	void end_frame(VkCommandBuffer cmd)
	{
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { image_available_semaphores_[current_frame_] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmd;

		VkSemaphore signalSemaphores[] = { render_finished_semaphores_[image_index_] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphics_queue_, 1, &submitInfo, in_flight_fences_[current_frame_]) != VK_SUCCESS) {
			logger_.log("Failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapchains[] = { vkb_swapchain_.swapchain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapchains;
		presentInfo.pImageIndices = &image_index_;

		vkQueuePresentKHR(present_queue_, &presentInfo);

		current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
	}

private:
	static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
		VkDebugUtilsMessageTypeFlagsEXT message_type,
		const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
		void* user_data)
	{
		Logger* logger = static_cast<Logger*>(user_data);
		if (logger)
		{
			std::string prefix = "[Vulkan Validation] ";
			if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
			{
				prefix += "ERROR: ";
			}
			else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
			{
				prefix += "WARNING: ";
			}

			logger->log(prefix + callback_data->pMessage);
		}
		return VK_FALSE;
	}

	Logger& logger_;
	Window& window_;

	vkb::Instance vkb_instance_;
	vkb::PhysicalDevice vkb_physical_device_;
	vkb::Device vkb_device_;
	vkb::Swapchain vkb_swapchain_;

	VkSurfaceKHR surface_ = VK_NULL_HANDLE;
	VkQueue graphics_queue_ = VK_NULL_HANDLE;
	VkQueue present_queue_ = VK_NULL_HANDLE;
	std::uint32_t graphics_queue_family_ = 0;

	std::vector<VkImage> swapchain_images_;
	std::vector<VkImageView> swapchain_image_views_;

	VmaAllocator allocator_ = VK_NULL_HANDLE;
	VkCommandPool command_pool_ = VK_NULL_HANDLE;

	VkSemaphore image_available_semaphores_[MAX_FRAMES_IN_FLIGHT]{};
	std::vector<VkSemaphore> render_finished_semaphores_;
	VkFence in_flight_fences_[MAX_FRAMES_IN_FLIGHT]{};

	std::uint32_t current_frame_ = 0;
	std::uint32_t image_index_ = 0;
};