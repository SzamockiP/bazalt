#pragma once
#include <atomic>
#include <optional>
#include <concepts>
#include <new>
#include <memory>

template <std::move_constructible T>
class MpscQueue
{
public:
    MpscQueue()
    {
        auto* stub = new Node{};
        _head.store(stub, std::memory_order_relaxed);
        _tail.store(stub, std::memory_order_relaxed);
    }

    ~MpscQueue() noexcept
    {
        Node* current = _head.load(std::memory_order_relaxed);
        while (current != nullptr)
        {
            Node* next = current->next.load(std::memory_order_relaxed);
            delete current;
            current = next;
        }
    }

    MpscQueue(const MpscQueue&) = delete;
    MpscQueue& operator=(const MpscQueue&) = delete;
    MpscQueue(MpscQueue&&) = delete;
    MpscQueue& operator=(MpscQueue&&) = delete;

    template <typename... Args>
    requires std::constructible_from<T, Args...>
    void emplace(Args&&... args)
    {
        auto node = std::make_unique<Node>();
        node->value.emplace(std::forward<Args>(args)...);
        Node* raw = node.release();

        Node* prev_tail = _tail.exchange(raw, std::memory_order_acq_rel);
        prev_tail->next.store(raw, std::memory_order_release);
    }

    void push(T&& value)
    {
        emplace(std::move(value));
    }

    void push(const T& value)
    {
        emplace(value);
    }

    [[nodiscard]] std::optional<T> pop() noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        Node* head_copy = _head.load(std::memory_order_relaxed);
        Node* next = head_copy->next.load(std::memory_order_acquire);

        if (next == nullptr)
            return std::nullopt;

        _head.store(next, std::memory_order_relaxed);
        std::optional<T> result = std::move(next->value);
        delete head_copy;
        return result;
    }

private:
    struct Node
    {
        std::atomic<Node*> next{nullptr};
        std::optional<T> value;
    };

    alignas(std::hardware_destructive_interference_size) std::atomic<Node*> _head;
    alignas(std::hardware_destructive_interference_size) std::atomic<Node*> _tail;
};