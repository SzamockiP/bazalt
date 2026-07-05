import lumapy as lp
import time
import collections

class App:
    engine = lp.Engine()

    @engine.onError
    def on_error(self, msg: str):
        print(msg)

    @engine.onFrame
    def on_update(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_count += 1
        self.fps_timer += dt
        
        if self.fps_timer >= 1.0:
            avg_fps = self.frame_count / self.fps_timer
            self.engine.setTitle(f"LumaPy Demo - Triangle | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            self.frame_count = 0
            self.fps_timer = 0.0

        self.engine.submit(self.cmd)

    def __init__(self):
        self.last_time = time.time()
        self.frame_count = 0
        self.fps_timer = 0.0
        self.engine.init(1024, 720, "LumaPy Demo - Triangle")

        vert_spv = self.engine.compileShader("triangle.vert", lp.ShaderStage.VERTEX)
        frag_spv = self.engine.compileShader("triangle.frag", lp.ShaderStage.FRAGMENT)

        self.pipeline = (self.engine.createPipeline()
            .vertexShader(vert_spv)
            .fragmentShader(frag_spv)
            .vertexFormat([lp.Format.FLOAT3, lp.Format.FLOAT3])
            .build())

        # Format: pos x, y, z, color r, g, b
        vertices = [
             0.0, -0.5, 0.0,   1.0, 0.0, 0.0,
            -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,
             0.5,  0.5, 0.0,   0.0, 0.0, 1.0,
        ]
        self.vbuf = self.engine.createBuffer(vertices, lp.BufferType.VERTEX, lp.DataType.FLOAT)

        indices = [0, 1, 2]
        self.ibuf = self.engine.createBuffer(indices, lp.BufferType.INDEX, lp.DataType.UINT32)

        self.cmd = self.engine.createCommandBuffer()

        # Nagrywamy liste komend tylko RAZ!
        self.cmd.begin()
        self.cmd.beginRendering(clear_color=[0.1, 0.2, 0.3, 1.0])
        self.cmd.setViewport()
        self.cmd.setScissor()
        self.cmd.bindPipeline(self.pipeline)
        self.cmd.bindVertexBuffer(self.vbuf)
        self.cmd.bindIndexBuffer(self.ibuf)
        self.cmd.drawIndexed(3)
        self.cmd.endRendering()

    def run(self):
        self.engine.run(self)

if __name__ == "__main__":
    app = App()
    app.run()
