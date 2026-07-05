#pragma once
#include <volk.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <string>
#include <stdexcept>
#include <memory>
#include <expected>

struct WindowDeleter
{
    void operator()(GLFWwindow* ptr) const noexcept
    {
        if (ptr)
        {
            glfwDestroyWindow(ptr);
        }
    };
} ;

struct MouseState
{
    float dx = 0.0f;
    float dy = 0.0f;
    float last_x = 0.0f;
    float last_y = 0.0f;
    bool first_mouse = true;
};

class Window
{
private:
    Window(int width, int height, const std::string& title) :
        width_(width), height_(height), title_(title) {}

public:
    static std::expected<std::unique_ptr<Window>, std::string> create(int width, int height, const std::string& title)
    {
        if (!glfwInit())
        {
            return std::unexpected("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        GLFWwindow* raw_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (!raw_window)
        {
            glfwTerminate();
            return std::unexpected("Failed to create window");
        }

        auto window = std::unique_ptr<Window>(new Window(width, height, title));
        window->window_.reset(raw_window);

        glfwSetWindowUserPointer(window->window_.get(), window.get());
        glfwSetCursorPosCallback(window->window_.get(), mouse_callback);

        glfwSetInputMode(window->window_.get(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        return window;
    };

    ~Window()
    {
        glfwTerminate();
    };

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool should_close() const
    {
        return glfwWindowShouldClose(window_.get());
    }

    void poll_events()
    {
        glfwPollEvents();
    }

    bool is_key_pressed(int key) const
    {
        return glfwGetKey(window_.get(), key) == GLFW_PRESS;
    }

    MouseState get_mouse_state() const
    {
        return mouse_;
    }

    GLFWwindow* get_native_handle() const { return window_.get(); }

    void set_title(const std::string& title)
    {
        title_ = title;
        glfwSetWindowTitle(window_.get(), title_.c_str());
    }

private:
    std::unique_ptr<GLFWwindow, WindowDeleter> window_;
    int width_;
    int height_;
    std::string title_;

    MouseState mouse_;


    static void mouse_callback(GLFWwindow* window, double xpos, double ypos)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (!win)
            return;

        float fx = static_cast<float>(xpos);
        float fy = static_cast<float>(ypos);

        if (win->mouse_.first_mouse)
        {
            win->mouse_.last_x = fx;
            win->mouse_.last_y = fy;
            win->mouse_.first_mouse = false;
        }

        win->mouse_.dx += fx - win->mouse_.last_x;
        win->mouse_.dy += win->mouse_.last_y - fy;

        win->mouse_.last_x = fx;
        win->mouse_.last_y = fy;
    };
};