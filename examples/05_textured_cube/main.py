import bazalt as bz
import glm
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
        self.engine.init(1024, 720, "Bazalt Demo - Textured Multi-Cube")
        self.engine.setCursorMode(bz.CURSOR_DISABLED)

        self.camera = Camera(pos=(0.0, 0.0, 3.0), speed=2.5)
        self.last_mouse_dx = 0.0
        self.last_mouse_dy = 0.0
        self.last_time = time.time()
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
        vert_spv = self.engine.compileShader("cube_tex.vert", bz.ShaderStage.VERTEX)
        frag_spv = self.engine.compileShader("cube_tex.frag", bz.ShaderStage.FRAGMENT)

        # Create a graphics pipeline using a builder pattern
        self.pipeline = (self.engine.createPipeline()
            .vertexShader(vert_spv)
            .fragmentShader(frag_spv)
            .vertexFormat([bz.Format.FLOAT3, bz.Format.FLOAT2]) # pos, uv
            .depthTest(True)
            .uniformBuffer(0, bz.ShaderStage.VERTEX, set=0)
            .texture(0, bz.ShaderStage.FRAGMENT, set=1)
            .build())

    def setup_buffers(self):
        # Format: pos x, y, z, uv u, v
        vertices = [
            # Front face (Bricks)
            -0.5, -0.5,  0.5,   0.0, 0.0,
             0.5, -0.5,  0.5,   1.0, 0.0,
             0.5,  0.5,  0.5,   1.0, 1.0,
            -0.5,  0.5,  0.5,   0.0, 1.0,
            # Back face (Bricks)
            -0.5, -0.5, -0.5,   1.0, 0.0,
             0.5, -0.5, -0.5,   0.0, 0.0,
             0.5,  0.5, -0.5,   0.0, 1.0,
            -0.5,  0.5, -0.5,   1.0, 1.0,
            # Left face (Crate)
            -0.5, -0.5, -0.5,   0.0, 0.0,
            -0.5, -0.5,  0.5,   1.0, 0.0,
            -0.5,  0.5,  0.5,   1.0, 1.0,
            -0.5,  0.5, -0.5,   0.0, 1.0,
            # Right face (Crate)
             0.5, -0.5, -0.5,   1.0, 0.0,
             0.5, -0.5,  0.5,   0.0, 0.0,
             0.5,  0.5,  0.5,   0.0, 1.0,
             0.5,  0.5, -0.5,   1.0, 1.0,
            # Top face (Bricks)
            -0.5, -0.5, -0.5,   0.0, 1.0,
             0.5, -0.5, -0.5,   1.0, 1.0,
             0.5, -0.5,  0.5,   1.0, 0.0,
            -0.5, -0.5,  0.5,   0.0, 0.0,
            # Bottom face (Crate)
            -0.5,  0.5, -0.5,   0.0, 0.0,
             0.5,  0.5, -0.5,   1.0, 0.0,
             0.5,  0.5,  0.5,   1.0, 1.0,
            -0.5,  0.5,  0.5,   0.0, 1.0,
        ]
        self.vbuf = self.engine.createBuffer(vertices, bz.BufferType.VERTEX, bz.DataType.FLOAT)

        # Reordered indices to group faces by texture
        indices = [
            # --- BRICKS ---
            # Front
            0, 1, 2, 2, 3, 0,
            # Back
            5, 4, 7, 7, 6, 5,
            # Top
            16, 17, 18, 18, 19, 16,
            
            # --- CRATE ---
            # Left
            8, 9, 10, 10, 11, 8,
            # Right
            13, 12, 15, 15, 14, 13,
            # Bottom
            23, 22, 21, 21, 20, 23
        ]
        self.ibuf = self.engine.createBuffer(indices, bz.BufferType.INDEX, bz.DataType.UINT32)

        self.ubuf = self.engine.createBuffer([0.0]*16, bz.BufferType.UNIFORM, bz.DataType.FLOAT)

    def setup_descriptors(self):
        tex1 = self.engine.loadTexture("../assets/wall.png")
        tex2 = self.engine.loadTexture("../assets/container.png")

        # Create Descriptor Pool and allocate descriptor sets
        self.pool = self.engine.createDescriptorPool(max_sets=4, samplers=2, uniform_buffers=2)
        
        # Set 0 (Frame set): 1 UBO
        self.frame_set = self.pool.allocateFrameDescriptorSet(self.pipeline, set=0)
        self.frame_set.setBuffer(0, self.ubuf)
        
        # Set 1 (Material set for Bricks)
        self.bricks_set = self.pool.allocateDescriptorSet(self.pipeline, set=1)
        self.bricks_set.setTexture(0, tex1)

        # Set 1 (Material set for Crate)
        self.crate_set = self.pool.allocateDescriptorSet(self.pipeline, set=1)
        self.crate_set.setTexture(0, tex2)

    def record_commands(self):
        # Create and record a command buffer
        self.cmd = self.engine.createCommandBuffer()
        self.cmd.begin()
        self.cmd.beginRendering(clear_color=[0.1, 0.2, 0.3, 1.0])
        self.cmd.setViewport()
        self.cmd.setScissor()
        self.cmd.bindPipeline(self.pipeline)
        
        # Bind global state (Camera)
        self.cmd.bindDescriptorSet(self.frame_set, self.pipeline, set=0)
        
        # Bind geometry
        self.cmd.bindVertexBuffer(self.vbuf)
        self.cmd.bindIndexBuffer(self.ibuf)
        
        # Draw first part: Bricks (18 indices)
        self.cmd.bindDescriptorSet(self.bricks_set, self.pipeline, set=1)
        self.cmd.drawIndexed(18, firstIndex=0)
        
        # Draw second part: Crate (18 indices)
        self.cmd.bindDescriptorSet(self.crate_set, self.pipeline, set=1)
        self.cmd.drawIndexed(18, firstIndex=18)
        
        self.cmd.endRendering()

    def on_update(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.frame_count += 1
        self.fps_timer += dt

        if self.fps_timer >= 1.0:
            avg_fps = self.frame_count / self.fps_timer
            self.engine.setTitle(f"Bazalt Demo - Textured Multi-Cube | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
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
