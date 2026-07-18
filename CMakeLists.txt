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
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stb)

add_library(_core MODULE
    src/main.cpp
    src/vma_impl.cpp
    src/stb_impl.cpp
    src/Buffer.hpp
    src/CommandBuffer.hpp
    src/Context.hpp
    src/DescriptorSet.hpp
    src/Error.hpp
    src/Features.hpp
    src/Format.hpp
    src/Image.hpp
    src/ImmediateSubmit.hpp
    src/Logger.hpp
    src/MpscQueue.hpp
    src/Pipeline.hpp
    src/Renderer.hpp
    src/RenderTarget.hpp
    src/Sampler.hpp
    src/ScopeGuard.hpp
    src/ShaderCompiler.hpp
    src/SurfaceProvider.hpp
    src/UploadManager.hpp
    src/window.hpp
)

target_link_libraries(_core PRIVATE 
    pybind11::module
    glfw
    volk
    vma
    vk-bootstrap::vk-bootstrap
    Vulkan::shaderc_combined
)
target_include_directories(_core PRIVATE ${stb_SOURCE_DIR})
set_target_properties(_core PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_compile_features(_core PRIVATE cxx_std_23)
if(MSVC)
    target_compile_options(_core PRIVATE /Zc:__cplusplus)
else()
    target_compile_options(_core PRIVATE -Wno-interference-size)
endif()

if(WIN32)
    target_compile_definitions(_core PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

install(TARGETS _core DESTINATION bazalt)