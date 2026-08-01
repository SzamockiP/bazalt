import sys
import os
import time
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QTimer
import bazalt as bz

class VulkanWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bazalt PyQt6 Integration Demo")
        self.resize(800, 600)
        
        # 1. Initialize Bazalt Logger and register callbacks
        self.logger = bz.Logger()
        @self.logger.on_message
        def on_message(msg):
            print(f"[{msg.severity}] {msg.text}")

        # 2. Initialize Bazalt Context and SwapchainRenderer (connected directly to native Win32 window handle)
        self.ctx = bz.Context(self.logger)
        self.renderer = self.ctx.create_renderer(win32_hwnd=int(self.winId()))

        self.setup_vulkan()
        
        # 3. Timer to tick frames asynchronously (integrates with Qt's event loop)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)  # ~60 FPS

        # FPS shown in the Qt window title, like the GLFW examples.
        self._last_time = time.time()
        self._frame_count = 0
        self._fps_timer = 0.0

    def setup_vulkan(self):
        # Paths to shaders
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vert_path = os.path.join(script_dir, "triangle.vert")
        frag_path = os.path.join(script_dir, "triangle.frag")
        
        # Compile GLSL shaders
        vert_spv = self.ctx.compile_shader(vert_path, bz.ShaderStage.VERTEX)
        frag_spv = self.ctx.compile_shader(frag_path, bz.ShaderStage.FRAGMENT)
        
        # Create pipeline: Position + Color
        self.pipeline = (self.ctx.graphics_pipeline()
            .vertex_shader(vert_spv)
            .fragment_shader(frag_spv)
            .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
            .build(self.renderer))
            
        # Triangle geometry
        vertices = [
             0.0, -0.5, 0.0,   1.0, 0.0, 0.0, # Top / Red
            -0.5,  0.5, 0.0,   0.0, 1.0, 0.0, # Bottom-Left / Green
             0.5,  0.5, 0.0,   0.0, 0.0, 1.0, # Bottom-Right / Blue
        ]
        self.vbuf = self.ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
        
        self.ibuf = self.ctx.create_buffer([0, 1, 2], bz.BufferType.INDEX, bz.MemoryUsage.STATIC, bz.DataType.UINT32)
        
        # Record commands
        self.cmd = self.ctx.create_command_buffer()
        self.cmd.begin()
        with self.cmd.rendering(self.renderer, clear_color=[0.15, 0.15, 0.2, 1.0]) as c:
            (c.bind_pipeline(self.pipeline)
              .bind_vertex_buffer(self.vbuf)
              .bind_index_buffer(self.ibuf)
              .draw_indexed(3))
        
    def tick(self):
        # begin_frame opens the logical frame; acquire takes this window's
        # swapchain image and handles automatic resize recreation
        self.ctx.begin_frame()
        if self.renderer.acquire():
            now = time.time()
            self._fps_timer += now - self._last_time
            self._last_time = now
            self._frame_count += 1
            if self._fps_timer >= 1.0:
                fps = self._frame_count / self._fps_timer
                self.setWindowTitle(
                    f"Bazalt PyQt6 Integration Demo | {1000.0 / fps:.2f} ms/frame | {fps:.1f} FPS")
                self._frame_count = 0
                self._fps_timer = 0.0
            self.renderer.present(self.cmd)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = VulkanWidget()
    widget.show()
    sys.exit(app.exec())
