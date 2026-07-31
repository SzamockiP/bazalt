#include "Bindings.hpp"

void bind_enums(py::module_& m)
{
    // py::arithmetic() so `msg.severity >= bz.Severity.WARNING` works — filtering
    // by level is the whole point of carrying one.
    py::enum_<Severity>(m, "Severity", py::arithmetic())
        .value("INFO", Severity::Info)
        .value("WARNING", Severity::Warning)
        .value("ERROR", Severity::Error);

    py::enum_<Source>(m, "Source")
        .value("GENERAL", Source::General)
        .value("VALIDATION", Source::Validation)
        .value("WINDOW", Source::Window)
        .value("SHADER", Source::Shader)
        .value("UPLOAD", Source::Upload)
        .value("DEVICE", Source::Device);

    // Capabilities, not versions/extensions: the same capability has different
    // spellings per driver (dynamic rendering is an extension on 1.2, core in
    // 1.3), so which one to use is bazalt's problem, not the user's. New entries
    // here are additive, so nothing about this needs to wait for a 2.0.
    py::enum_<Feature>(m, "Feature")
        .value("ANISOTROPIC_FILTERING", Feature::ANISOTROPIC_FILTERING)
        .value("WIREFRAME", Feature::WIREFRAME)
        .value("WIDE_LINES", Feature::WIDE_LINES)
        .value("DEPTH_CLAMP", Feature::DEPTH_CLAMP)
        .value("SAMPLE_RATE_SHADING", Feature::SAMPLE_RATE_SHADING)
        .value("MULTI_DRAW_INDIRECT", Feature::MULTI_DRAW_INDIRECT)
        .value("SHADER_FLOAT64", Feature::SHADER_FLOAT64)
        .value("INDEPENDENT_BLEND", Feature::INDEPENDENT_BLEND)
        .value("TESSELLATION", Feature::TESSELLATION)
        .value("GEOMETRY_SHADER", Feature::GEOMETRY_SHADER)
        .value("FRAGMENT_STORES", Feature::FRAGMENT_STORES)
        .value("VERTEX_STAGE_STORES", Feature::VERTEX_STAGE_STORES)
        .value("MULTIVIEW", Feature::MULTIVIEW)
        .value("BINDLESS", Feature::BINDLESS)
        .value("DRAW_INDIRECT_COUNT", Feature::DRAW_INDIRECT_COUNT)
        // The three below answer True on every full Vulkan driver, because they name
        // restrictions only a portability subset applies. See Features.hpp.
        .value("COMPARISON_SAMPLER", Feature::COMPARISON_SAMPLER)
        .value("SAMPLER_MIP_LOD_BIAS", Feature::SAMPLER_MIP_LOD_BIAS)
        .value("MULTISAMPLE_ARRAYS", Feature::MULTISAMPLE_ARRAYS);

    // The gamepad layout GLFW maps every known pad onto, renamed rather than
    // translated: the values ARE the GLFW ones, so the two cannot drift.
    py::enum_<GamepadButton>(m, "GamepadButton")
        .value("A", GamepadButton::A)
        .value("B", GamepadButton::B)
        .value("X", GamepadButton::X)
        .value("Y", GamepadButton::Y)
        .value("LEFT_BUMPER", GamepadButton::LEFT_BUMPER)
        .value("RIGHT_BUMPER", GamepadButton::RIGHT_BUMPER)
        .value("BACK", GamepadButton::BACK)
        .value("START", GamepadButton::START)
        .value("GUIDE", GamepadButton::GUIDE)
        .value("LEFT_THUMB", GamepadButton::LEFT_THUMB)
        .value("RIGHT_THUMB", GamepadButton::RIGHT_THUMB)
        .value("DPAD_UP", GamepadButton::DPAD_UP)
        .value("DPAD_RIGHT", GamepadButton::DPAD_RIGHT)
        .value("DPAD_DOWN", GamepadButton::DPAD_DOWN)
        .value("DPAD_LEFT", GamepadButton::DPAD_LEFT);

    py::enum_<GamepadAxis>(m, "GamepadAxis")
        .value("LEFT_X", GamepadAxis::LEFT_X)
        .value("LEFT_Y", GamepadAxis::LEFT_Y)
        .value("RIGHT_X", GamepadAxis::RIGHT_X)
        .value("RIGHT_Y", GamepadAxis::RIGHT_Y)
        .value("LEFT_TRIGGER", GamepadAxis::LEFT_TRIGGER)
        .value("RIGHT_TRIGGER", GamepadAxis::RIGHT_TRIGGER);

    py::enum_<BufferType>(m, "BufferType")
        .value("VERTEX", BufferType::VERTEX)
        .value("INDEX", BufferType::INDEX)
        .value("UNIFORM", BufferType::UNIFORM)
        .value("STORAGE", BufferType::STORAGE);

    py::enum_<DataType>(m, "DataType")
        .value("FLOAT", DataType::FLOAT)
        .value("UINT32", DataType::UINT32)
        .value("UINT16", DataType::UINT16)
        .value("INT32", DataType::INT32);

    py::enum_<ShaderStage>(m, "ShaderStage")
        .value("VERTEX", ShaderStage::VERTEX)
        .value("FRAGMENT", ShaderStage::FRAGMENT)
        .value("COMPUTE", ShaderStage::COMPUTE)
        .value("TESS_CONTROL", ShaderStage::TESS_CONTROL)
        .value("TESS_EVALUATION", ShaderStage::TESS_EVALUATION)
        .value("GEOMETRY", ShaderStage::GEOMETRY);

    py::enum_<VertexFormat>(m, "VertexFormat")
        .value("FLOAT2", VertexFormat::FLOAT2)
        .value("FLOAT3", VertexFormat::FLOAT3)
        .value("FLOAT4", VertexFormat::FLOAT4)
        .value("FLOAT", VertexFormat::FLOAT)
        .value("UBYTE4_NORM", VertexFormat::UBYTE4_NORM)
        .value("UINT", VertexFormat::UINT);

    py::enum_<Topology>(m, "Topology")
        .value("TRIANGLE_LIST", Topology::TRIANGLE_LIST)
        .value("POINT_LIST", Topology::POINT_LIST)
        .value("LINE_LIST", Topology::LINE_LIST)
        .value("TRIANGLE_STRIP", Topology::TRIANGLE_STRIP)
        .value("LINE_STRIP", Topology::LINE_STRIP)
        .value("PATCH_LIST", Topology::PATCH_LIST);

    // The vocabulary of cmd.barrier() in manual mode (auto_barriers=False).
    py::enum_<Access>(m, "Access")
        .value("SHADER_READ", Access::SHADER_READ)
        .value("SHADER_WRITE", Access::SHADER_WRITE)
        .value("VERTEX_READ", Access::VERTEX_READ)
        .value("INDEX_READ", Access::INDEX_READ)
        .value("UNIFORM_READ", Access::UNIFORM_READ)
        .value("INDIRECT_READ", Access::INDIRECT_READ);

    // Pixel formats — the name VertexFormat freed in 0.4.
    py::enum_<Format>(m, "Format")
        .value("RGBA8", Format::RGBA8)
        .value("RGBA8_SRGB", Format::RGBA8_SRGB)
        .value("BGRA8", Format::BGRA8)
        .value("R8", Format::R8)
        .value("RG8", Format::RG8)
        .value("R16F", Format::R16F)
        .value("RGBA16F", Format::RGBA16F)
        .value("R32F", Format::R32F)
        .value("RGBA32F", Format::RGBA32F)
        .value("D32F", Format::D32F)
        .value("R32_UINT", Format::R32_UINT)
        .value("R11G11B10F", Format::R11G11B10F)
        .value("DEPTH_STENCIL", Format::DEPTH_STENCIL);

    py::enum_<Filter>(m, "Filter").value("LINEAR", Filter::LINEAR).value("NEAREST", Filter::NEAREST);

    py::enum_<AddressMode>(m, "AddressMode")
        .value("REPEAT", AddressMode::REPEAT)
        .value("CLAMP", AddressMode::CLAMP)
        .value("MIRROR", AddressMode::MIRROR)
        .value("CLAMP_TO_BORDER", AddressMode::CLAMP_TO_BORDER);

    py::enum_<BorderColor>(m, "BorderColor")
        .value("OPAQUE_BLACK", BorderColor::OPAQUE_BLACK)
        .value("OPAQUE_WHITE", BorderColor::OPAQUE_WHITE);

    py::enum_<CompareOp>(m, "CompareOp")
        .value("NEVER", CompareOp::NEVER)
        .value("LESS", CompareOp::LESS)
        .value("EQUAL", CompareOp::EQUAL)
        .value("LESS_OR_EQUAL", CompareOp::LESS_OR_EQUAL)
        .value("GREATER", CompareOp::GREATER)
        .value("NOT_EQUAL", CompareOp::NOT_EQUAL)
        .value("GREATER_OR_EQUAL", CompareOp::GREATER_OR_EQUAL)
        .value("ALWAYS", CompareOp::ALWAYS);

    py::enum_<StencilOp>(m, "StencilOp")
        .value("KEEP", StencilOp::KEEP)
        .value("ZERO", StencilOp::ZERO)
        .value("REPLACE", StencilOp::REPLACE)
        .value("INCREMENT_CLAMP", StencilOp::INCREMENT_CLAMP)
        .value("DECREMENT_CLAMP", StencilOp::DECREMENT_CLAMP)
        .value("INVERT", StencilOp::INVERT)
        .value("INCREMENT_WRAP", StencilOp::INCREMENT_WRAP)
        .value("DECREMENT_WRAP", StencilOp::DECREMENT_WRAP);

    py::enum_<BlendMode>(m, "BlendMode")
        .value("ALPHA", BlendMode::ALPHA)
        .value("ADDITIVE", BlendMode::ADDITIVE)
        .value("PREMULTIPLIED", BlendMode::PREMULTIPLIED);

    py::enum_<PolygonMode>(m, "PolygonMode")
        .value("FILL", PolygonMode::FILL)
        .value("LINE", PolygonMode::LINE)
        .value("POINT", PolygonMode::POINT);

    py::enum_<CullMode>(m, "CullMode")
        .value("NONE", CullMode::NONE)
        .value("BACK", CullMode::BACK)
        .value("FRONT", CullMode::FRONT)
        .value("FRONT_AND_BACK", CullMode::FRONT_AND_BACK);

    py::enum_<FrontFace>(m, "FrontFace")
        .value("CLOCKWISE", FrontFace::CLOCKWISE)
        .value("COUNTER_CLOCKWISE", FrontFace::COUNTER_CLOCKWISE);

    py::enum_<MemoryUsage>(m, "MemoryUsage")
        .value("STATIC", MemoryUsage::STATIC)
        .value("DYNAMIC", MemoryUsage::DYNAMIC);

    // ── Window (GLFW) ──
    py::enum_<WindowMode>(m, "WindowMode")
        .value("WINDOWED", WindowMode::WINDOWED)
        .value("FRAMELESS", WindowMode::FRAMELESS)
        .value("FULLSCREEN", WindowMode::FULLSCREEN)
        .value("FULLSCREEN_WINDOWED", WindowMode::FULLSCREEN_WINDOWED);

    // ── SwapchainRenderer ──
    // Inherits RenderTarget: presenting to a window is one way to consume a
    // rendered image, not the definition of rendering.
    py::enum_<PresentMode>(m, "PresentMode")
        .value("FIFO", PresentMode::FIFO)
        .value("MAILBOX", PresentMode::MAILBOX)
        .value("IMMEDIATE", PresentMode::IMMEDIATE)
        .value("FIFO_RELAXED", PresentMode::FIFO_RELAXED);
}
