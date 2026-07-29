#pragma once
// Precompiled header: third-party and standard headers ONLY.
//
// The eight binding translation units each parse pybind11, volk and the C++23
// standard library before they reach a line of bazalt code. Parsing that once
// and reusing it is what makes the split pay: without this, splitting one big
// translation unit into eight mostly multiplies the fixed cost by eight.
//
// No src/*.hpp goes in here, and this was measured rather than reasoned. Putting
// the core headers in as well moves a rebuild after touching one binding file from
// 25.8s to 20.8s, and a rebuild after touching a core header from 43.9s to 45.4s,
// because the PCH itself then has to be rebuilt first and that step is serial.
// Roughly a wash, so the version that keeps the PCH valid across every header edit
// wins on the simpler behaviour.

// pybind11 FIRST, and the order is load-bearing. volk.h includes <windows.h> when
// VK_USE_PLATFORM_WIN32_KHR is defined, <windows.h> defines min and max as macros,
// and pybind11/numpy.h calls std::min -- which then fails to compile. main.cpp had
// this order by luck until 0.20; here it is on purpose.
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
