cmake_minimum_required(VERSION 3.28)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
# GLFW
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

add_library(lumapy MODULE src/main.cpp src/MpscQueue.hpp src/Logger.hpp src/window.hpp )
target_link_libraries(lumapy PRIVATE pybind11::module glfw)
set_target_properties(lumapy PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_compile_features(lumapy PRIVATE cxx_std_23)
if(MSVC)
    target_compile_options(lumapy PRIVATE /Zc:__cplusplus)
endif()

install(TARGETS lumapy DESTINATION .)