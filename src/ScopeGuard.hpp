#pragma once
#include <utility>

// Runs the stored callable on scope exit unless release()d first.
//
// Multi-step Vulkan construction creates objects that must all be destroyed
// when a later step fails. Without a guard every failure branch repeats the
// same cleanup loop — PipelineBuilder::build carried four copies of one.
template <typename F>
class ScopeGuard
{
public:
    explicit ScopeGuard(F fn)
        : fn_(std::move(fn))
    {
    }

    ~ScopeGuard()
    {
        if (armed_)
        {
            fn_();
        }
    }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    // Call once the guarded objects have found an owner.
    void release()
    {
        armed_ = false;
    }

private:
    F fn_;
    bool armed_ = true;
};
