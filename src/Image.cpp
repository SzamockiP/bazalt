#include "Image.hpp"

void record_image_transition(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkAccessFlags srcAccess,
    VkAccessFlags dstAccess,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage,
    VkImageAspectFlags aspect,
    std::uint32_t baseMip,
    std::uint32_t mipCount,
    std::uint32_t layerCount,
    std::uint32_t baseLayer)
{
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = srcAccess,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = aspect,
            .baseMipLevel = baseMip,
            .levelCount = mipCount,
            .baseArrayLayer = baseLayer,
            .layerCount = layerCount}};
    vk.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// ── Image ─────────────────────────────────────────────────────────────────────

Image::Image(
    std::shared_ptr<Context> context,
    VkImage image,
    VmaAllocation allocation,
    VkImageView view,
    Format format,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t mip_levels,
    std::uint32_t array_layers,
    bool cube,
    VkImageView storage_view,
    VkSampleCountFlagBits samples,
    std::uint32_t depth)
    : context_(std::move(context)),
      image_(image),
      allocation_(allocation),
      view_(view),
      storage_view_(storage_view),
      format_(format),
      width_(width),
      height_(height),
      depth_(depth),
      mip_levels_(mip_levels),
      array_layers_(array_layers),
      cube_(cube),
      samples_(samples)
{
    // The layout state needs the geometry to index a split; it starts
    // uniform-UNDEFINED, so nothing is allocated until something writes a
    // strict subset.
    layouts_.configure(array_layers, mip_levels);
}

Image::~Image()
{
    if (context_)
    {
        context_->defer_destroy(
            [vk = &context_->vk(),
             device = context_->device(),
             allocator = context_->allocator(),
             view = view_,
             storage_view = storage_view_,
             image = image_,
             allocation = allocation_]
            {
                if (view != VK_NULL_HANDLE)
                {
                    vk->vkDestroyImageView(device, view, nullptr);
                }
                // A separate 2D_ARRAY view exists only for cubemaps (storage
                // binding can't use a CUBE view); non-cube images share view_.
                if (storage_view != VK_NULL_HANDLE)
                {
                    vk->vkDestroyImageView(device, storage_view, nullptr);
                }
                if (image != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
                {
                    vmaDestroyImage(allocator, image, allocation);
                }
            });
    }
}

void Image::mark_subresource_contents(
    VkImageLayout layout,
    std::uint32_t base_layer,
    std::uint32_t layer_count,
    std::uint32_t base_mip,
    std::uint32_t mip_count)
{
    // A caller quoting a barrier range may say VK_REMAINING_ARRAY_LAYERS
    // (a 3D slice target does); the layout state wants the real count.
    layer_count = (std::ranges::min)(layer_count, array_layers_ - base_layer);
    layouts_.set_range(layout, base_layer, layer_count, base_mip, mip_count);
    has_contents_.store(true);
}

bool Image::ready() const
{
    switch (upload_state_.load())
    {
        case UploadState::None:
            return has_contents_.load();
        // Both mean "not usable": one will be, one never will, and wait()
        // is what tells them apart.
        case UploadState::Pending:
        case UploadState::Failed:
            return false;
        case UploadState::Submitted:
            return context_->completed_submit_serial() >= upload_serial_.load();
    }
    return false;
}

std::expected<void, Error> Image::wait()
{
    if (auto r = wait_submitted_(); !r)
    {
        return std::unexpected(r.error());
    }
    if (upload_state_.load() == UploadState::Submitted)
    {
        return context_->wait_for_serial(upload_serial_.load());
    }
    return {};
}

std::expected<std::uint64_t, Error> Image::require_resident()
{
    if (upload_state_.load() == UploadState::None)
    {
        return 0;
    }
    if (auto r = wait_submitted_(); !r)
    {
        return std::unexpected(r.error());
    }
    return upload_serial_.load();
}

void Image::set_upload_pending()
{
    {
        std::lock_guard lock(upload_mutex_);
        ++pending_uploads_;
        upload_state_.store(UploadState::Pending);
    }
    upload_cv_.notify_all();
}

void Image::set_upload_submitted(std::uint64_t serial)
{
    {
        std::lock_guard lock(upload_mutex_);
        upload_serial_.store(serial);
        has_contents_.store(true);
        // Zero already, and legitimately so, for create_image(array): that
        // path submits inline on the calling thread and never queues a job,
        // so there is nothing outstanding to retire.
        if (pending_uploads_ > 0)
        {
            --pending_uploads_;
        }
        // Still Pending while anything is queued behind this one. The serial
        // above is the newest submitted, so a waiter that gets through waits
        // for the last job rather than for whichever finished first.
        if (pending_uploads_ == 0 && upload_state_.load() != UploadState::Failed)
        {
            upload_state_.store(UploadState::Submitted);
        }
    }
    upload_cv_.notify_all();
}

void Image::set_upload_failed(std::string message)
{
    {
        std::lock_guard lock(upload_mutex_);
        upload_error_ = std::move(message);
        // Retires this job like a submit does, so a waiter is not left
        // counting one that will never arrive. Failed wins over Submitted
        // while anything is outstanding: a later job succeeding does not make
        // the earlier failure untrue, and the next set_upload_pending clears
        // it, which is what it did before it could be one job of several.
        if (pending_uploads_ > 0)
        {
            --pending_uploads_;
        }
        upload_state_.store(UploadState::Failed);
    }
    upload_cv_.notify_all();
}

// ── Creation ──────────────────────────────────────────────────────────────────

VkImageUsageFlags Image::usage_for(Context& context, Format format)
{
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(context.physical_device(), context.vk_format(format), &props);
    const VkFormatFeatureFlags feat = props.optimalTilingFeatures;

    VkImageUsageFlags usage = 0;
    if (feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
    {
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (feat & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT)
    {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    if (feat & VK_FORMAT_FEATURE_TRANSFER_DST_BIT)
    {
        usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }
    if (feat & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
    {
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if (feat & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
    {
        usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    if (feat & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
    {
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    return usage;
}

VkImageUsageFlags Image::usage_for_image(Context& context, Format format, VkSampleCountFlagBits samples)
{
    VkImageUsageFlags usage = usage_for(context, format);
    if (has_stencil(context.vk_format(format)))
    {
        usage &= ~static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    }
    if (samples != VK_SAMPLE_COUNT_1_BIT)
    {
        usage &= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    return usage;
}

bool Image::can_generate_mips(Context& context, Format format)
{
    if (format_info(format).depth)
    {
        return false;
    }
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(context.physical_device(), context.vk_format(format), &props);
    constexpr VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (props.optimalTilingFeatures & needed) == needed;
}

bool Image::can_blit(Context& context, Format from, Format to)
{
    VkFormatProperties src_props{};
    vkGetPhysicalDeviceFormatProperties(context.physical_device(), context.vk_format(from), &src_props);
    VkFormatProperties dst_props{};
    vkGetPhysicalDeviceFormatProperties(context.physical_device(), context.vk_format(to), &dst_props);

    constexpr VkFormatFeatureFlags src_needed = VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (src_props.optimalTilingFeatures & src_needed) == src_needed &&
           (dst_props.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
}

std::expected<std::shared_ptr<Image>, Error> Image::create_empty(
    Context& context,
    std::uint32_t width,
    std::uint32_t height,
    Format format,
    std::uint32_t mip_levels,
    std::uint32_t array_layers,
    bool cube,
    VkSampleCountFlagBits samples,
    std::uint32_t depth)
{
    if (width == 0 || height == 0 || depth == 0)
    {
        return std::unexpected(
            err_resource(std::format("Image dimensions must be non-zero, got {}x{}x{}", width, height, depth)));
    }
    // Vulkan's own rule: a 3D image has exactly one array layer, so a
    // layered or cube volume is not a combination that exists. Multisampled
    // 3D images do not exist either (VUID-VkImageCreateInfo-samples-02257).
    if (depth > 1 && (array_layers > 1 || cube))
    {
        return std::unexpected(err_resource(
            "A 3D image (depth>1) cannot have layers or be a cubemap: Vulkan gives a "
            "volume exactly one array layer. Use depth= alone, or layers=/cube= on a "
            "2D image."));
    }
    if (depth > 1 && samples != VK_SAMPLE_COUNT_1_BIT)
    {
        return std::unexpected(err_resource("A 3D image (depth>1) cannot be multisampled"));
    }
    // A multisampled image is an MSAA attachment and nothing else. It can be a
    // layered attachment (MSAA render-to-layer resolves per layer), but it can't
    // carry mips (no blitting between sample counts) or be a cubemap (you sample
    // the single-sample resolve, never a multisampled cube view). Fail here
    // rather than at vkCreateImage.
    if (samples != VK_SAMPLE_COUNT_1_BIT && (mip_levels != 1 || cube))
    {
        return std::unexpected(err_resource(
            "A multisampled image (samples>1) is a render-target attachment only: "
            "it cannot have mipmaps or be a cubemap"));
    }
    // Layered MSAA is plain Vulkan and a portability driver may still refuse it
    // (Metal has no multisampled texture array). One feature to ask, rather than
    // a validation error at vkCreateImage (0.22).
    if (samples != VK_SAMPLE_COUNT_1_BIT && array_layers > 1 && !context.supports(Feature::MULTISAMPLE_ARRAYS))
    {
        return std::unexpected(err_unsupported(
            "A multisampled image with layers>1 needs the MULTISAMPLE_ARRAYS feature, which this driver "
            "does not offer. Ask ctx.supports(bz.Feature.MULTISAMPLE_ARRAYS), or render the layers one "
            "at a time into single-layer multisampled targets."));
    }
    const VkFormat vk_fmt = context.vk_format(format);
    if (vk_fmt == VK_FORMAT_UNDEFINED)
    {
        return std::unexpected(err_unsupported(std::format("This device supports no {} format", format_name(format))));
    }
    const VkImageAspectFlags aspect = aspect_mask_for(vk_fmt);
    // A volume has one array layer and a cube has six, so the three cases are
    // exclusive and the order only decides which name a reader sees first.
    const VkImageViewType view_type = [&]
    {
        if (depth > 1)
        {
            return VK_IMAGE_VIEW_TYPE_3D;
        }
        if (cube)
        {
            return VK_IMAGE_VIEW_TYPE_CUBE;
        }
        return array_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }();

    // 2D_ARRAY_COMPATIBLE is what later lets a 2D view select one Z slice
    // of the volume as a render-target attachment. It costs nothing where
    // legal, and a portability driver that lacks imageView2DOn3DImage would
    // reject the flag itself — so it is set only where the feature answers.
    VkImageCreateFlags flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    if (depth > 1 && context.supports(Feature::IMAGE_VIEW_2D_ON_3D))
    {
        flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    }

    VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .imageType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
        .format = vk_fmt,
        .extent = {width, height, depth},
        .mipLevels = mip_levels,
        .arrayLayers = array_layers,
        .samples = samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        // A multisampled attachment is only rendered into and resolved out, so
        // it keeps just the attachment usage: STORAGE on a multisample image
        // needs a feature we don't enable, and SAMPLED/TRANSFER are dead weight
        // (you sample the single-sample resolve, never this).
        .usage = usage_for_image(context, format, samples),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    if (auto e = check(
            vmaCreateImage(context.allocator(), &imageInfo, &allocInfo, &image, &allocation, nullptr),
            std::format("create {} image", format_name(format)),
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }

    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = image,
        .viewType = view_type,
        .format = vk_fmt,
        .components = {},
        .subresourceRange = {aspect, 0, mip_levels, 0, array_layers}};

    VkImageView view = VK_NULL_HANDLE;
    if (auto e = check(
            context.vk().vkCreateImageView(context.device(), &viewInfo, nullptr, &view),
            std::format("create {} image view", format_name(format)),
            ErrorCode::Resource))
    {
        vmaDestroyImage(context.allocator(), image, allocation);
        return std::unexpected(*e);
    }

    // Storage images may not be bound through a CUBE view. Give a cubemap a
    // parallel 2D_ARRAY view so compute can write its faces (procedural
    // skyboxes); sampling still goes through the CUBE view above.
    VkImageView storage_view = VK_NULL_HANDLE;
    if (cube)
    {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        if (auto e = check(
                context.vk().vkCreateImageView(context.device(), &viewInfo, nullptr, &storage_view),
                std::format("create {} storage view", format_name(format)),
                ErrorCode::Resource))
        {
            context.vk().vkDestroyImageView(context.device(), view, nullptr);
            vmaDestroyImage(context.allocator(), image, allocation);
            return std::unexpected(*e);
        }
    }

    return std::make_shared<Image>(
        context.shared_from_this(),
        image,
        allocation,
        view,
        format,
        width,
        height,
        mip_levels,
        array_layers,
        cube,
        storage_view,
        samples,
        depth);
}

std::expected<std::shared_ptr<Image>, Error> Image::create_from_pixels(
    Context& context,
    const void* pixels,
    std::uint32_t width,
    std::uint32_t height,
    Format format,
    bool mipmaps,
    std::uint32_t depth)
{
    const std::uint32_t mips = mipmaps && can_generate_mips(context, format) ? full_mip_count(width, height, depth) : 1;
    auto image = create_empty(context, width, height, format, mips, 1, false, VK_SAMPLE_COUNT_1_BIT, depth);
    if (!image)
    {
        return image;
    }
    if (auto r = (*image)->upload_pixels(context, pixels, mips); !r)
    {
        return std::unexpected(r.error());
    }
    return image;
}

std::expected<std::shared_ptr<Image>, Error> Image::create_layered_from_pixels(
    Context& context,
    const void* pixels,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t layers,
    bool cube,
    Format format,
    bool mipmaps)
{
    const std::uint32_t mips = mipmaps && can_generate_mips(context, format) ? full_mip_count(width, height) : 1;
    auto image = create_empty(context, width, height, format, mips, layers, cube);
    if (!image)
    {
        return image;
    }
    if (auto r = (*image)->upload_pixels(context, pixels, mips); !r)
    {
        return std::unexpected(r.error());
    }
    return image;
}

// ── Readback ──────────────────────────────────────────────────────────────────

std::expected<std::vector<std::byte>, Error> Image::read(bool all_layers, std::uint32_t layer, std::uint32_t mip)
{
    const std::uint32_t layers = all_layers ? array_layers_ : 1;
    const std::uint32_t base_layer = all_layers ? 0 : layer;
    // A pending async upload is finished first — read() is blocking anyway.
    if (auto w = wait(); !w)
    {
        return std::unexpected(w.error());
    }
    if (!has_contents_.load())
    {
        return std::unexpected(err_resource(
            "read() called on an Image that has no contents yet. Upload to it or "
            "render into it first"));
    }
    if (samples_ != VK_SAMPLE_COUNT_1_BIT)
    {
        return std::unexpected(err_resource(
            "read() called on a multisampled image. Read the target's resolved "
            "single-sample attachment (target.color[i] / target.depth) instead"));
    }
    if (!fits_within(base_layer, layers, array_layers_))
    {
        return std::unexpected(
            err_resource(std::format("read(layer={}): this image has {} layer(s)", base_layer, array_layers_)));
    }
    if (mip >= mip_levels_)
    {
        return std::unexpected(
            err_resource(std::format("read(mip={}): this image has {} mip level(s)", mip, mip_levels_)));
    }

    const FormatInfo info = format_info(format_);
    // A packed or combined format has no single numpy dtype: one pixel of
    // DEPTH_STENCIL is depth AND stencil, one pixel of R11G11B10F is three
    // channels sharing 32 bits. Refuse here with the reason rather than hand
    // back bytes that mean nothing.
    if (info.numpy_dtype[0] == '\0')
    {
        return std::unexpected(err_resource(
            std::format(
                "read() is not available for {}: the format packs several values into one "
                "texel, so there is no array shape that describes it. Render it into an "
                "RGBA target, or use a shader to unpack it.",
                format_name(format_))));
    }
    const std::uint32_t mip_width = mip_extent(width_, mip);
    const std::uint32_t mip_height = mip_extent(height_, mip);
    const std::uint32_t mip_depth = mip_extent(depth_, mip);
    const VkDeviceSize size = static_cast<VkDeviceSize>(mip_width) * mip_height * mip_depth * info.bytes_per_pixel *
                              layers;
    const VkImageAspectFlags aspect = aspect_mask_for(vk_format());

    auto staging_pair = create_staging_buffer(*context_, size, Staging::Readback);
    if (!staging_pair)
    {
        return std::unexpected(staging_pair.error());
    }
    auto [staging, staging_alloc] = *staging_pair;

    auto submitted = immediate_submit(
        *context_,
        [&](VkCommandBuffer cmd)
        {
            // Exactly the subresources being read, each from the layout it
            // is actually in. A single barrier over the whole image can only
            // name one oldLayout, and naming the wrong one is a validation
            // error plus undefined contents — which is precisely what
            // reading a partially rendered image used to do.
            for_each_subresource_(
                base_layer,
                layers,
                mip,
                [&](std::uint32_t l, std::uint32_t m)
                {
                    record_image_transition(
                        context_->vk(),
                        cmd,
                        image_,
                        layouts_.get(l, m),
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_ACCESS_MEMORY_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        aspect,
                        m,
                        1,
                        barrier_layers(1),
                        l);
                });

            VkBufferImageCopy region{
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {aspect, mip, base_layer, layers},
                .imageOffset = {0, 0, 0},
                .imageExtent = {mip_width, mip_height, mip_depth}};
            context_->vk().vkCmdCopyImageToBuffer(
                cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);

            // …and put each one back where it was, which is what makes
            // read() non-destructive on a partially written image.
            for_each_subresource_(
                base_layer,
                layers,
                mip,
                [&](std::uint32_t l, std::uint32_t m)
                {
                    record_image_transition(
                        context_->vk(),
                        cmd,
                        image_,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        layouts_.get(l, m),
                        VK_ACCESS_TRANSFER_READ_BIT,
                        VK_ACCESS_MEMORY_READ_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        aspect,
                        m,
                        1,
                        barrier_layers(1),
                        l);
                });
        });
    if (!submitted)
    {
        vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
        return std::unexpected(submitted.error());
    }

    std::vector<std::byte> out(static_cast<std::size_t>(size));
    void* mapped = nullptr;
    if (auto e = check(
            vmaMapMemory(context_->allocator(), staging_alloc, &mapped),
            "map readback buffer memory",
            ErrorCode::Resource))
    {
        vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
        return std::unexpected(*e);
    }
    std::memcpy(out.data(), mapped, static_cast<std::size_t>(size));
    vmaUnmapMemory(context_->allocator(), staging_alloc);
    vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);

    return out;
}

// ── Recorded uploads and mip generation ───────────────────────────────────────

void Image::record_upload_commands(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips)
{
    record_image_transition(
        context_->vk(),
        cmd,
        image_,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0,
        mips,
        barrier_layers(array_layers_));

    record_copy_and_finalize_(cmd, staging, mips);
}

void Image::record_reload_commands(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips)
{
    record_image_transition(
        context_->vk(),
        cmd,
        image_,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0,
        mips,
        barrier_layers(array_layers_));

    record_copy_and_finalize_(cmd, staging, mips);
}

void Image::record_update_commands(
    VkCommandBuffer cmd,
    VkBuffer staging,
    std::uint32_t layer,
    std::uint32_t mip,
    VkOffset3D offset,
    VkExtent3D extent,
    VkImageLayout from)
{
    const VkImageAspectFlags aspect = aspect_mask_for(vk_format());
    const VkPipelineStageFlags shader_stages = context_->all_shader_stages();
    record_image_transition(
        context_->vk(),
        cmd,
        image_,
        from,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        aspect,
        mip,
        1,
        barrier_layers(1),
        layer);

    VkBufferImageCopy region{
        .bufferOffset = 0,
        // Zero means "tightly packed to imageExtent", which is what the
        // staging buffer is: the binding copies the caller's rectangle into
        // it row by row, so there is no source padding to describe.
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {aspect, mip, layer, 1},
        .imageOffset = offset,
        .imageExtent = extent};
    context_->vk().vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    record_image_transition(
        context_->vk(),
        cmd,
        image_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        shader_stages,
        aspect,
        mip,
        1,
        barrier_layers(1),
        layer);
}

void Image::record_generate_mipmaps(
    VkCommandBuffer cmd,
    VkImageLayout src_layout,
    VkPipelineStageFlags src_stage,
    VkAccessFlags src_access)
{
    // mip 0 -> TRANSFER_SRC (this transition is also the barrier waiting on
    // the producer of mip 0).
    record_image_transition(
        context_->vk(),
        cmd,
        image_,
        src_layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        src_access,
        VK_ACCESS_TRANSFER_READ_BIT,
        src_stage,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0,
        1,
        barrier_layers(array_layers_));

    // The other levels hold nothing worth keeping -> discard into TRANSFER_DST.
    record_image_transition(
        context_->vk(),
        cmd,
        image_,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        1,
        mip_levels_ - 1,
        barrier_layers(array_layers_));

    auto mip_width = static_cast<std::int32_t>(width_);
    auto mip_height = static_cast<std::int32_t>(height_);
    // Depth halves alongside width and height for a volume (it is 1 and
    // stays 1 for everything else), so the loop terminates exactly where
    // full_mip_count says the chain ends.
    auto mip_depth = static_cast<std::int32_t>(depth_);

    for (std::uint32_t i = 1; i < mip_levels_; ++i)
    {
        const std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
        const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;
        const std::int32_t next_depth = mip_depth > 1 ? mip_depth / 2 : 1;

        // One blit with layerCount = array_layers_ fills level i for every
        // face/layer at once (they share mip dimensions).
        VkImageBlit blit{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, array_layers_},
            .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, mip_depth}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, array_layers_},
            .dstOffsets = {{0, 0, 0}, {next_width, next_height, next_depth}}};
        context_->vk().vkCmdBlitImage(
            cmd,
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_LINEAR);

        // Level i-1 is done being read from -> retire to SHADER_READ_ONLY.
        record_image_transition(
            context_->vk(),
            cmd,
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            i - 1,
            1,
            barrier_layers(array_layers_));

        if (i + 1 < mip_levels_)
        {
            // Level i becomes the source for the next blit.
            record_image_transition(
                context_->vk(),
                cmd,
                image_,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                i,
                1,
                barrier_layers(array_layers_));
        }
        else
        {
            // The last level retires straight to SHADER_READ_ONLY.
            record_image_transition(
                context_->vk(),
                cmd,
                image_,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                i,
                1,
                barrier_layers(array_layers_));
        }

        mip_width = next_width;
        mip_height = next_height;
        mip_depth = next_depth;
    }
}

std::expected<std::pair<VkBuffer, VmaAllocation>, Error> Image::create_filled_staging(
    Context& context,
    const void* pixels)
{
    const FormatInfo info = format_info(format_);
    // Layers and depth are never both >1 (create_empty refuses it), so one
    // multiply chain covers the array case and the volume case.
    const VkDeviceSize size = static_cast<VkDeviceSize>(width_) * height_ * info.bytes_per_pixel * array_layers_ *
                              depth_;
    return create_staging_buffer(context, size, Staging::Upload, pixels);
}

std::expected<void, Error> Image::wait_submitted_()
{
    if (upload_state_.load() == UploadState::Pending || upload_state_.load() == UploadState::Failed)
    {
        std::unique_lock lock(upload_mutex_);
        // Every queued job, not just the first to be submitted. The predicate
        // used to be "the state is no longer Pending", which one worker
        // submitting the first of several satisfies immediately.
        upload_cv_.wait(lock, [&] { return pending_uploads_ == 0; });
        if (upload_state_.load() == UploadState::Failed)
        {
            return std::unexpected(err_resource(upload_error_));
        }
    }
    return {};
}

std::expected<void, Error> Image::upload_pixels(Context& context, const void* pixels, std::uint32_t mips)
{
    auto staging = create_filled_staging(context, pixels);
    if (!staging)
    {
        return std::unexpected(staging.error());
    }
    auto [buffer, allocation] = *staging;

    auto serial = deferred_submit(context, [&](VkCommandBuffer cmd) { record_upload_commands(cmd, buffer, mips); });
    if (!serial)
    {
        vmaDestroyBuffer(context.allocator(), buffer, allocation);
        return std::unexpected(serial.error());
    }
    // The GPU reads the staging buffer after this returns, so it retires on
    // the serial rather than here.
    context.defer_destroy([allocator = context.allocator(), buffer, allocation]
                          { vmaDestroyBuffer(allocator, buffer, allocation); });

    mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    set_upload_submitted(*serial);
    context.note_upload_serial(*serial);
    return {};
}

void Image::record_copy_and_finalize_(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips)
{
    // One region with layerCount = array_layers_ copies every layer: the
    // staging buffer holds them consecutively, exactly what a layered copy
    // reads from bufferOffset 0.
    VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, array_layers_},
        .imageOffset = {0, 0, 0},
        .imageExtent = {width_, height_, depth_}};
    context_->vk().vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    if (mips > 1)
    {
        record_mip_generation(context_->vk(), cmd, image_, width_, height_, mips, array_layers_, depth_);
    }
    else
    {
        record_image_transition(
            context_->vk(),
            cmd,
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            barrier_layers(array_layers_));
    }
}

void Image::record_mip_generation(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    VkImage image,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t mips,
    std::uint32_t layers,
    std::uint32_t depth)
{
    auto mip_width = static_cast<std::int32_t>(width);
    auto mip_height = static_cast<std::int32_t>(height);
    auto mip_depth = static_cast<std::int32_t>(depth);
    // Barriers on a volume name VK_REMAINING_ARRAY_LAYERS; see barrier_layers.
    const std::uint32_t barrier_span = depth > 1 ? VK_REMAINING_ARRAY_LAYERS : layers;

    // Every layer shares mip dimensions, so one blit with layerCount = layers
    // generates the whole level for all faces/layers at once.
    for (std::uint32_t i = 1; i < mips; ++i)
    {
        record_image_transition(
            vk,
            cmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            i - 1,
            1,
            barrier_span);

        const std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
        const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;
        const std::int32_t next_depth = mip_depth > 1 ? mip_depth / 2 : 1;

        VkImageBlit blit{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, layers},
            .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, mip_depth}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, layers},
            .dstOffsets = {{0, 0, 0}, {next_width, next_height, next_depth}}};
        vk.vkCmdBlitImage(
            cmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_LINEAR);

        record_image_transition(
            vk,
            cmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            i - 1,
            1,
            barrier_span);

        mip_width = next_width;
        mip_height = next_height;
        mip_depth = next_depth;
    }

    record_image_transition(
        vk,
        cmd,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        mips - 1,
        1,
        barrier_span);
}

// ── Image-to-image recorders ──────────────────────────────────────────────────

void record_image_copy(const VolkDeviceTable& vk, VkCommandBuffer cmd, Image& src, Image& dst, VkImageLayout src_layout)
{
    // Read off the source rather than threaded in as a parameter: both images
    // already belong to this Context (the binding layer compares owners), so the
    // fact is here, and record_image_transition takes `vk` for the same reason.
    const VkPipelineStageFlags shader_stages = src.owner()->all_shader_stages();
    const std::uint32_t layers = src.array_layers();
    const std::uint32_t barrier_span = src.barrier_layers(layers);
    // Every level the two images share. 0.17 copied mip 0 only and called the
    // rest a ceiling ("a full chain is N regions for a case that has not come
    // up"). The case came up: a copy that leaves levels 1..N holding the
    // destination's old pixels is not a copy of the image, it is a copy of its
    // top level, and the difference shows up the moment anything samples with a
    // mip bias. N regions in one vkCmdCopyImage is the same one call.
    const std::uint32_t mips = (std::ranges::min)(src.mip_levels(), dst.mip_levels());

    record_image_transition(
        vk,
        cmd,
        src.vk_image(),
        src_layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        src.aspect(),
        0,
        mips,
        barrier_span);
    // The destination is overwritten in full, so its old contents are discarded
    // rather than waited for — UNDEFINED is legal from any layout.
    record_image_transition(
        vk,
        cmd,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        dst.aspect(),
        0,
        mips,
        barrier_span);

    std::vector<VkImageCopy> regions;
    regions.reserve(mips);
    for (std::uint32_t mip = 0; mip < mips; ++mip)
    {
        // Level dimensions floor at 1, which is what the mip chain of a
        // non-square image does on its short axis before the long one. Depth
        // participates for a volume and is 1 everywhere else.
        const std::uint32_t w = (std::ranges::max)(src.width() >> mip, 1u);
        const std::uint32_t h = (std::ranges::max)(src.height() >> mip, 1u);
        const std::uint32_t d = (std::ranges::max)(src.depth() >> mip, 1u);
        regions.push_back(
            VkImageCopy{
                .srcSubresource =
                    {.aspectMask = src.aspect(), .mipLevel = mip, .baseArrayLayer = 0, .layerCount = layers},
                .srcOffset = {0, 0, 0},
                .dstSubresource =
                    {.aspectMask = dst.aspect(), .mipLevel = mip, .baseArrayLayer = 0, .layerCount = layers},
                .dstOffset = {0, 0, 0},
                .extent = {w, h, d}});
    }
    vk.vkCmdCopyImage(
        cmd,
        src.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        static_cast<std::uint32_t>(regions.size()),
        regions.data());

    for (Image* image : {&src, &dst})
    {
        record_image_transition(
            vk,
            cmd,
            image->vk_image(),
            image == &src ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            image == &src ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            shader_stages,
            image->aspect(),
            0,
            mips,
            barrier_span);
    }
}

void record_image_blit(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& src,
    Image& dst,
    VkImageLayout src_layout,
    VkFilter filter)
{
    const VkPipelineStageFlags shader_stages = src.owner()->all_shader_stages();
    const std::uint32_t layers = (std::ranges::min)(src.array_layers(), dst.array_layers());
    const std::uint32_t barrier_span = src.barrier_layers(layers);

    record_image_transition(
        vk,
        cmd,
        src.vk_image(),
        src_layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        src.aspect(),
        0,
        1,
        barrier_span);
    record_image_transition(
        vk,
        cmd,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        dst.aspect(),
        0,
        1,
        barrier_span);

    VkImageBlit region{
        .srcSubresource = {.aspectMask = src.aspect(), .mipLevel = 0, .baseArrayLayer = 0, .layerCount = layers},
        .srcOffsets =
            {{0, 0, 0},
             {static_cast<std::int32_t>(src.width()),
              static_cast<std::int32_t>(src.height()),
              static_cast<std::int32_t>(src.depth())}},
        .dstSubresource = {.aspectMask = dst.aspect(), .mipLevel = 0, .baseArrayLayer = 0, .layerCount = layers},
        .dstOffsets = {
            {0, 0, 0},
            {static_cast<std::int32_t>(dst.width()),
             static_cast<std::int32_t>(dst.height()),
             static_cast<std::int32_t>(dst.depth())}}};
    vk.vkCmdBlitImage(
        cmd,
        src.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region,
        filter);

    for (Image* image : {&src, &dst})
    {
        record_image_transition(
            vk,
            cmd,
            image->vk_image(),
            image == &src ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            image == &src ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            shader_stages,
            image->aspect(),
            0,
            1,
            barrier_span);
    }
}

void record_image_clear(const VolkDeviceTable& vk, VkCommandBuffer cmd, Image& image, std::array<float, 4> color)
{
    const VkPipelineStageFlags shader_stages = image.owner()->all_shader_stages();
    const std::uint32_t barrier_span = image.barrier_layers(image.array_layers());
    record_image_transition(
        vk,
        cmd,
        image.vk_image(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        image.aspect(),
        0,
        image.mip_levels(),
        barrier_span);

    const VkClearColorValue value{{color[0], color[1], color[2], color[3]}};
    VkImageSubresourceRange range{
        .aspectMask = image.aspect(),
        .baseMipLevel = 0,
        .levelCount = image.mip_levels(),
        .baseArrayLayer = 0,
        .layerCount = barrier_span};
    vk.vkCmdClearColorImage(cmd, image.vk_image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);

    record_image_transition(
        vk,
        cmd,
        image.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        shader_stages,
        image.aspect(),
        0,
        image.mip_levels(),
        barrier_span);
}
