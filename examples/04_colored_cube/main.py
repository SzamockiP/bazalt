import bazalt as bz
import glm
import numpy as np
import time
import math

class Camera:
    def __init__(self, pos=(0.0, 0.0, 3.0), yaw=-math.pi/2, speed=2.5):
        self.pos = glm.vec3(*pos)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = yaw
        self.pitch = 0.0
        self.sensitivity = 0.002
        self.speed = speed

    def update_mouse(self, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.pitch + dy * self.sensitivity))
        
        front = glm.vec3(
            math.cos(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            math.sin(self.yaw) * math.cos(self.pitch)
        )
        self.front = glm.normalize(front)
        
        right = glm.normalize(glm.cross(self.front, glm.vec3(0.0, 1.0, 0.0)))
        self.up = glm.normalize(glm.cross(right, self.front))
        return right

    def process_keyboard(self, engine, dt, right):
        velocity = self.speed * dt
        if engine.isKeyPressed(bz.KEY_W): self.pos += velocity * self.front
        if engine.isKeyPressed(bz.KEY_S): self.pos -= velocity * self.front
        if engine.isKeyPressed(bz.KEY_A): self.pos -= velocity * right
        if engine.isKeyPressed(bz.KEY_D): self.pos += velocity * right
        if engine.isKeyPressed(bz.KEY_SPACE): self.pos += velocity * self.up
        if engine.isKeyPressed(bz.KEY_LEFT_SHIFT): self.pos -= velocity * self.up

    def get_matrices(self, aspect_ratio):
        view = glm.lookAt(self.pos, self.pos + self.front, self.up)
        proj = glm.perspectiveRH_ZO(glm.radians(45.0), aspect_ratio, 0.1, 100.0)
        proj[1][1] *= -1 
        model = glm.mat4(1.0)
        return view, proj, model

class App:
    def __init__(self):
        # 1. Initialize the engine
        self.engine = bz.Engine()
        # 2. Create a window (width, height, title)
        self.engine.init(1024, 720, "Bazalt Demo - 3D Cube")
        self.engine.setCursorMode(bz.CURSOR_DISABLED)

        self.camera = Camera(pos=(0.0, 0.0, 3.0), speed=2.5)
        self.last_time = time.time()
        self.last_mouse_dx = 0.0
        self.last_mouse_dy = 0.0
        self.frame_count = 0
        self.fps_timer = 0.0

        self.setup_pipeline()
        self.setup_buffers()
        self.setup_descriptors()
        self.record_commands()

        self.engine.onError(self.on_error)
        self.engine.onFrame(self.on_update)

    def on_error(self, msg):
        print(msg)

    def setup_pipeline(self):
        # Compile GLSL shaders
        vert_spv = self.engine.compileShader("cube.vert", bz.ShaderStage.VERTEX)
        frag_spv = self.engine.compileShader("cube.frag", bz.ShaderStage.FRAGMENT)

        # Create a graphics pipeline using a builder pattern
        self.pipeline = (self.engine.createPipeline()
            .vertexShader(vert_spv)
            .fragmentShader(frag_spv)
            .vertexFormat([bz.Format.FLOAT3, bz.Format.FLOAT3])
            .depthTest(True)
            .cullMode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE)
            .blend(False)
            .uniformBuffer(0, bz.ShaderStage.VERTEX, set=0)
            .build())

    def setup_buffers(self):
        # Format: pos x, y, z, normal nx, ny, nz
        vertices = np.array([
            # Front face (z = 0.5) normal = (0, 0, 1)
            -0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
             0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
             0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
            -0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
            # Back face (z = -0.5) normal = (0, 0, -1)
            -0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
             0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
             0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
            -0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
            # Left face (x = -0.5) normal = (-1, 0, 0)
            -0.5, -0.5, -0.5,  -1.0, 0.0, 0.0,
            -0.5, -0.5,  0.5,  -1.0, 0.0, 0.0,
            -0.5,  0.5,  0.5,  -1.0, 0.0, 0.0,
            -0.5,  0.5, -0.5,  -1.0, 0.0, 0.0,
            # Right face (x = 0.5) normal = (1, 0, 0)
             0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
             0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
             0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
             0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
            # Top face (y = -0.5) normal = (0, -1, 0)
            -0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
             0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
             0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
            -0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
            # Bottom face (y = 0.5) normal = (0, 1, 0)
            -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
             0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
             0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
            -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
        ], dtype=np.float32)
        self.vbuf = self.engine.createBuffer(vertices, bz.BufferType.VERTEX)

        indices = np.array([
            0, 1, 2, 2, 3, 0,       # Front
            5, 4, 7, 7, 6, 5,       # Back
            8, 9, 10, 10, 11, 8,    # Left
            13, 12, 15, 15, 14, 13, # Right
            16, 17, 18, 18, 19, 16, # Top
            23, 22, 21, 21, 20, 23  # Bottom
        ], dtype=np.uint32)
        self.ibuf = self.engine.createBuffer(indices, bz.BufferType.INDEX)

        self.ubuf = self.engine.createBuffer(np.zeros(16, dtype=np.float32), bz.BufferType.UNIFORM)

    def setup_descriptors(self):
        # Create Descriptor Pool and allocate descriptor set for uniform buffer
        self.pool = self.engine.createDescriptorPool(max_sets=2, uniform_buffers=2)
        self.desc_set = self.pool.allocateFrameDescriptorSet(self.pipeline, set=0)
        self.desc_set.setBuffer(0, self.ubuf)

    def record_commands(self):
        # Create and record a command buffer
        self.cmd = self.engine.createCommandBuffer()
        self.cmd.begin()
        self.cmd.beginRendering(clear_color=[0.1, 0.2, 0.3, 1.0])
        self.cmd.setViewport()
        self.cmd.setScissor()
        # Bind resources
        self.cmd.bindPipeline(self.pipeline)
        self.cmd.bindDescriptorSet(self.desc_set, self.pipeline, set=0)
        self.cmd.bindVertexBuffer(self.vbuf)
        self.cmd.bindIndexBuffer(self.ibuf)
        
        # Draw 36 indices (12 triangles = 6 quads = 1 cube)
        self.cmd.drawIndexed(36)
        self.cmd.endRendering()

    def on_update(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.frame_count += 1
        self.fps_timer += dt
        if self.fps_timer >= 1.0:
            avg_fps = self.frame_count / self.fps_timer
            self.engine.setTitle(f"Bazalt Demo - 3D Cube | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            self.frame_count = 0
            self.fps_timer = 0.0

        mouse = self.engine.getMouseState()
        dx = mouse.dx - self.last_mouse_dx
        dy = mouse.dy - self.last_mouse_dy
        self.last_mouse_dx, self.last_mouse_dy = mouse.dx, mouse.dy

        right_vec = self.camera.update_mouse(dx, dy)
        self.camera.process_keyboard(self.engine, dt, right_vec)

        view, proj, model = self.camera.get_matrices(1024.0 / 720.0)
        mvp = proj * view * model
        
        self.ubuf.update(bytes(glm.transpose(mvp)))
        
        # Submit our pre-recorded command buffer to the GPU each frame
        self.engine.submit(self.cmd)

    def run(self):
        self.engine.run()

if __name__ == "__main__":
    app = App()
    app.run()
