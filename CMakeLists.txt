cmake_minimum_required(VERSION 3.28)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)

include(FetchContent)

# glfw
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

#volk
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 1.4.350
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(volk)
if(TARGET volk)
    set_target_properties(volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#vma
FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(vma)

if(NOT TARGET vma)
    add_library(vma INTERFACE)
    target_include_directories(vma INTERFACE ${vma_SOURCE_DIR}/include)
    target_compile_definitions(vma INTERFACE VMA_STATIC_VULKAN_FUNCTIONS=0 VMA_DYNAMIC_VULKAN_FUNCTIONS=1)
endif()

#vk-bootstrap
FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG v1.4.350
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(vk-bootstrap)
if(TARGET vk-bootstrap)
    set_target_properties(vk-bootstrap PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#stb
# A commit, not a branch. stb publishes no tags, so `master` used to be the only
# spelling available -- and it made an upstream push able to break a build here
# with no local change, which is the one failure nobody can reproduce. GitHub
# serves a shallow fetch of an explicit SHA, so pinning costs no clone time.
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stb)

# The binding layer is eight translation units, not one. Until 0.20 main.cpp
# carried every binding and included the whole header-only core, so a rebuild used
# one core out of however many the machine has. The headers stay header-only --
# they are templated and inline, and splitting them would be a rewrite for a cost
# the precompiled header already removes.
#
# The .hpp files used to be listed here for IDE display. They are gone because the
# list drifted: HotReload.hpp, Device.hpp, SpirvReflect.hpp, ResourceTracker.hpp
# and MpscQueue.hpp were all missing, and nothing noticed, which is what a list
# that serves no build purpose does.
add_library(_core MODULE
    src/main.cpp
    src/bindings/Enums.cpp
    src/bindings/Resources.cpp
    src/bindings/Pipelines.cpp
    src/bindings/Commands.cpp
    src/bindings/Windowing.cpp
    src/bindings/ContextBind.cpp
    src/bindings/Targets.cpp
    src/vma_impl.cpp
    src/stb_impl.cpp
)

target_link_libraries(_core PRIVATE 
    pybind11::module
    glfw
    volk
    vma
    vk-bootstrap::vk-bootstrap
    Vulkan::shaderc_combined
)
# src/ on the path because the binding files sit in src/bindings/ and include the
# core headers by name.
target_include_directories(_core PRIVATE ${stb_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Parse pybind11, volk and the C++23 standard library once instead of eight times.
target_precompile_headers(_core PRIVATE src/bindings/Pch.hpp)

# target_precompile_headers applies to EVERY source of the target, and these two
# exist only to instantiate a third-party implementation. Forcing pybind11 into
# them is waste at best, and vma_impl.cpp is order-sensitive around
# VMA_IMPLEMENTATION.
set_source_files_properties(src/vma_impl.cpp src/stb_impl.cpp
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
set_target_properties(_core PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

# VK_KHR_portability_subset lives in vulkan_beta.h, so its feature struct is
# invisible without this. Defined on every platform, not only Apple: the code that
# reads it is the same everywhere and only the runtime answer differs, and a struct
# that exists on one platform and not another is how you get a header that compiles
# in CI and not on a contributor's machine.
target_compile_definitions(_core PRIVATE VK_ENABLE_BETA_EXTENSIONS)

target_compile_features(_core PRIVATE cxx_std_23)
if(MSVC)
    # /bigobj: every pybind lambda is a template instantiation, and 0.14 pushed
    # the binding layer past COFF's 65,536-section limit. The 0.20 split into
    # eight files bought headroom, not immunity: any one of them can grow into it.
    #
    # /utf-8: the sources ARE UTF-8, but MSVC assumes the system code page for
    # both the source and the execution charset. Without this an em dash in an
    # error message reaches Python as a single 0x97 byte, which is not valid
    # UTF-8 — so the message arrives with a replacement character in it, or (when
    # it goes through std::format) empty. Eleven user-facing messages were
    # affected before 0.19. Telling the compiler the truth about the encoding is
    # the root fix; policing the prose for non-ASCII characters is not.
    target_compile_options(_core PRIVATE /Zc:__cplusplus /bigobj /utf-8)
else()
    target_compile_options(_core PRIVATE -Wno-interference-size)
endif()

if(WIN32)
    target_compile_definitions(_core PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

# The macOS mirror of the line above: it makes volk load vkCreateMetalSurfaceEXT,
# which is the surface entry point vk-bootstrap enables there. Nothing else about
# the platform needs code -- vk-bootstrap adds VK_KHR_portability_enumeration and
# the instance flag by itself, and enables VK_KHR_portability_subset on the device.
if(APPLE)
    target_compile_definitions(_core PRIVATE VK_USE_PLATFORM_METAL_EXT)

    # libc++ annotates <format> and the floating-point std::to_chars it calls with
    # availability, so a deployment target below 13.3 fails to compile every
    # binding file. Say so in one sentence instead of leaving the reader to find
    # it in a hundred lines of "'format' is unavailable".
    if(CMAKE_OSX_DEPLOYMENT_TARGET AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 13.3)
        message(FATAL_ERROR
            "bazalt needs a macOS deployment target of 13.3 or later, because libc++ "
            "supplies std::format from that version. Set MACOSX_DEPLOYMENT_TARGET=14.0 "
            "(the version the published wheels use).")
    endif()
endif()

install(TARGETS _core DESTINATION bazalt)