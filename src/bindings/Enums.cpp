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
        .value("MULTISAMPLE_ARRAYS", Feature::MULTISAMPLE_ARRAYS)
        .value("IMAGE_VIEW_2D_ON_3D", Feature::IMAGE_VIEW_2D_ON_3D);

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

    // The keyboard follows the gamepad's pattern (0.23): renamed, not
    // translated — the values ARE the GLFW ones. The bare KEY_* module ints
    // from before stay valid, because the query methods take int and an enum
    // member converts through its value.
    py::enum_<Key>(m, "Key")
        .value("SPACE", Key::SPACE)
        .value("APOSTROPHE", Key::APOSTROPHE)
        .value("COMMA", Key::COMMA)
        .value("MINUS", Key::MINUS)
        .value("PERIOD", Key::PERIOD)
        .value("SLASH", Key::SLASH)
        .value("D0", Key::D0)
        .value("D1", Key::D1)
        .value("D2", Key::D2)
        .value("D3", Key::D3)
        .value("D4", Key::D4)
        .value("D5", Key::D5)
        .value("D6", Key::D6)
        .value("D7", Key::D7)
        .value("D8", Key::D8)
        .value("D9", Key::D9)
        .value("SEMICOLON", Key::SEMICOLON)
        .value("EQUAL", Key::EQUAL)
        .value("A", Key::A)
        .value("B", Key::B)
        .value("C", Key::C)
        .value("D", Key::D)
        .value("E", Key::E)
        .value("F", Key::F)
        .value("G", Key::G)
        .value("H", Key::H)
        .value("I", Key::I)
        .value("J", Key::J)
        .value("K", Key::K)
        .value("L", Key::L)
        .value("M", Key::M)
        .value("N", Key::N)
        .value("O", Key::O)
        .value("P", Key::P)
        .value("Q", Key::Q)
        .value("R", Key::R)
        .value("S", Key::S)
        .value("T", Key::T)
        .value("U", Key::U)
        .value("V", Key::V)
        .value("W", Key::W)
        .value("X", Key::X)
        .value("Y", Key::Y)
        .value("Z", Key::Z)
        .value("LEFT_BRACKET", Key::LEFT_BRACKET)
        .value("BACKSLASH", Key::BACKSLASH)
        .value("RIGHT_BRACKET", Key::RIGHT_BRACKET)
        .value("GRAVE_ACCENT", Key::GRAVE_ACCENT)
        .value("WORLD_1", Key::WORLD_1)
        .value("WORLD_2", Key::WORLD_2)
        .value("ESCAPE", Key::ESCAPE)
        .value("ENTER", Key::ENTER)
        .value("TAB", Key::TAB)
        .value("BACKSPACE", Key::BACKSPACE)
        .value("INSERT", Key::INSERT)
        .value("DELETE", Key::DEL)
        .value("RIGHT", Key::RIGHT)
        .value("LEFT", Key::LEFT)
        .value("DOWN", Key::DOWN)
        .value("UP", Key::UP)
        .value("PAGE_UP", Key::PAGE_UP)
        .value("PAGE_DOWN", Key::PAGE_DOWN)
        .value("HOME", Key::HOME)
        .value("END", Key::END)
        .value("CAPS_LOCK", Key::CAPS_LOCK)
        .value("SCROLL_LOCK", Key::SCROLL_LOCK)
        .value("NUM_LOCK", Key::NUM_LOCK)
        .value("PRINT_SCREEN", Key::PRINT_SCREEN)
        .value("PAUSE", Key::PAUSE)
        .value("F1", Key::F1)
        .value("F2", Key::F2)
        .value("F3", Key::F3)
        .value("F4", Key::F4)
        .value("F5", Key::F5)
        .value("F6", Key::F6)
        .value("F7", Key::F7)
        .value("F8", Key::F8)
        .value("F9", Key::F9)
        .value("F10", Key::F10)
        .value("F11", Key::F11)
        .value("F12", Key::F12)
        .value("F13", Key::F13)
        .value("F14", Key::F14)
        .value("F15", Key::F15)
        .value("F16", Key::F16)
        .value("F17", Key::F17)
        .value("F18", Key::F18)
        .value("F19", Key::F19)
        .value("F20", Key::F20)
        .value("F21", Key::F21)
        .value("F22", Key::F22)
        .value("F23", Key::F23)
        .value("F24", Key::F24)
        .value("F25", Key::F25)
        .value("KP_0", Key::KP_0)
        .value("KP_1", Key::KP_1)
        .value("KP_2", Key::KP_2)
        .value("KP_3", Key::KP_3)
        .value("KP_4", Key::KP_4)
        .value("KP_5", Key::KP_5)
        .value("KP_6", Key::KP_6)
        .value("KP_7", Key::KP_7)
        .value("KP_8", Key::KP_8)
        .value("KP_9", Key::KP_9)
        .value("KP_DECIMAL", Key::KP_DECIMAL)
        .value("KP_DIVIDE", Key::KP_DIVIDE)
        .value("KP_MULTIPLY", Key::KP_MULTIPLY)
        .value("KP_SUBTRACT", Key::KP_SUBTRACT)
        .value("KP_ADD", Key::KP_ADD)
        .value("KP_ENTER", Key::KP_ENTER)
        .value("KP_EQUAL", Key::KP_EQUAL)
        .value("LEFT_SHIFT", Key::LEFT_SHIFT)
        .value("LEFT_CONTROL", Key::LEFT_CONTROL)
        .value("LEFT_ALT", Key::LEFT_ALT)
        .value("LEFT_SUPER", Key::LEFT_SUPER)
        .value("RIGHT_SHIFT", Key::RIGHT_SHIFT)
        .value("RIGHT_CONTROL", Key::RIGHT_CONTROL)
        .value("RIGHT_ALT", Key::RIGHT_ALT)
        .value("RIGHT_SUPER", Key::RIGHT_SUPER)
        .value("MENU", Key::MENU);

    py::enum_<MouseButton>(m, "MouseButton")
        .value("LEFT", MouseButton::LEFT)
        .value("RIGHT", MouseButton::RIGHT)
        .value("MIDDLE", MouseButton::MIDDLE)
        .value("BUTTON_4", MouseButton::BUTTON_4)
        .value("BUTTON_5", MouseButton::BUTTON_5)
        .value("BUTTON_6", MouseButton::BUTTON_6)
        .value("BUTTON_7", MouseButton::BUTTON_7)
        .value("BUTTON_8", MouseButton::BUTTON_8);

    py::enum_<CursorMode>(m, "CursorMode")
        .value("NORMAL", CursorMode::NORMAL)
        .value("HIDDEN", CursorMode::HIDDEN)
        .value("DISABLED", CursorMode::DISABLED);

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

    // Which parser reads the text. Two members and no SPIRV one: SPIR-V is a
    // compiled format rather than a language, and it already has its own two
    // spellings — a .spv path, or bytes in source=.
    py::enum_<ShaderLanguage>(m, "ShaderLanguage")
        .value("GLSL", ShaderLanguage::GLSL)
        .value("HLSL", ShaderLanguage::HLSL);

    py::enum_<VertexFormat>(m, "VertexFormat")
        .value("FLOAT2", VertexFormat::FLOAT2)
        .value("FLOAT3", VertexFormat::FLOAT3)
        .value("FLOAT4", VertexFormat::FLOAT4)
        .value("FLOAT", VertexFormat::FLOAT)
        .value("UBYTE4_NORM", VertexFormat::UBYTE4_NORM)
        .value("UINT", VertexFormat::UINT)
        .value("UINT2", VertexFormat::UINT2)
        .value("UINT3", VertexFormat::UINT3)
        .value("UINT4", VertexFormat::UINT4)
        .value("UBYTE4_UINT", VertexFormat::UBYTE4_UINT);

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
        .value("PREMULTIPLIED", BlendMode::PREMULTIPLIED)
        .value("MULTIPLY", BlendMode::MULTIPLY);

    // The axes the four modes are points in (0.23). No constant-colour or
    // dual-source factors: each needs more API than an enum row.
    py::enum_<BlendFactor>(m, "BlendFactor")
        .value("ZERO", BlendFactor::ZERO)
        .value("ONE", BlendFactor::ONE)
        .value("SRC_COLOR", BlendFactor::SRC_COLOR)
        .value("ONE_MINUS_SRC_COLOR", BlendFactor::ONE_MINUS_SRC_COLOR)
        .value("DST_COLOR", BlendFactor::DST_COLOR)
        .value("ONE_MINUS_DST_COLOR", BlendFactor::ONE_MINUS_DST_COLOR)
        .value("SRC_ALPHA", BlendFactor::SRC_ALPHA)
        .value("ONE_MINUS_SRC_ALPHA", BlendFactor::ONE_MINUS_SRC_ALPHA)
        .value("DST_ALPHA", BlendFactor::DST_ALPHA)
        .value("ONE_MINUS_DST_ALPHA", BlendFactor::ONE_MINUS_DST_ALPHA)
        .value("SRC_ALPHA_SATURATE", BlendFactor::SRC_ALPHA_SATURATE);

    py::enum_<BlendOp>(m, "BlendOp")
        .value("ADD", BlendOp::ADD)
        .value("SUBTRACT", BlendOp::SUBTRACT)
        .value("REVERSE_SUBTRACT", BlendOp::REVERSE_SUBTRACT)
        .value("MIN", BlendOp::MIN)
        .value("MAX", BlendOp::MAX);

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
