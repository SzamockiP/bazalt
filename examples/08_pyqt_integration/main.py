import sys
import os
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
        self.renderer = bz.SwapchainRenderer(win32_hwnd=int(self.winId()), context=self.ctx)

        self.setup_vulkan()
        
        # 3. Timer to tick frames asynchronously (integrates with Qt's event loop)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)  # ~60 FPS
        
    def setup_vulkan(self):
        # Paths to shaders
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vert_path = os.path.join(script_dir, "triangle.vert")
        frag_path = os.path.join(script_dir, "triangle.frag")
        
        # Compile GLSL shaders
        vert_spv = self.ctx.compile_shader(vert_path, bz.ShaderStage.VERTEX)
        frag_spv = self.ctx.compile_shader(frag_path, bz.ShaderStage.FRAGMENT)
        
        # Create pipeline: Position + Color
        self.pipeline = (self.ctx.pipeline_builder()
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
        self.cmd.begin_rendering(self.renderer, clear_color=[0.15, 0.15, 0.2, 1.0])
        self.cmd.bind_pipeline(self.pipeline)
        self.cmd.bind_vertex_buffer(self.vbuf)
        self.cmd.bind_index_buffer(self.ibuf)
        self.cmd.draw_indexed(3)
        self.cmd.end_rendering(self.renderer)
        
    def tick(self):
        # begin_frame acquires swapchain image and handles automatic resize recreation
        if frame := self.renderer.begin_frame():
            frame.submit(self.cmd)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = VulkanWidget()
    widget.show()
    sys.exit(app.exec())
