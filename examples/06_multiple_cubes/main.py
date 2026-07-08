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
        self.engine.init(1024, 720, "Bazalt Demo - 10 Cubes Lighting")
        self.engine.setCursorMode(bz.CURSOR_DISABLED)

        self.camera = Camera(pos=(0.0, 0.0, 5.0), speed=5.0)
        
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
        vert_spv = self.engine.compileShader("10cubes.vert", bz.ShaderStage.VERTEX)
        frag_spv = self.engine.compileShader("10cubes.frag", bz.ShaderStage.FRAGMENT)

        # Create a graphics pipeline using a builder pattern
        self.pipeline = (self.engine.createPipeline()
            .vertexShader(vert_spv)
            .fragmentShader(frag_spv)
            .vertexFormat([bz.Format.FLOAT3, bz.Format.FLOAT3, bz.Format.FLOAT3]) # pos, normal, color
            .depthTest(True)
            .uniformBuffer(0, bz.ShaderStage.VERTEX, set=0)   
            .uniformBuffer(0, bz.ShaderStage.FRAGMENT, set=0) 
            .build())

    def setup_buffers(self):
        # Format: pos x, y, z, normal x, y, z, color r, g, b
        vertices = [
            # Front face
            -0.5, -0.5,  0.5,   0.0,  0.0,  1.0,   1.0, 0.0, 0.0,
             0.5, -0.5,  0.5,   0.0,  0.0,  1.0,   1.0, 0.0, 0.0,
             0.5,  0.5,  0.5,   0.0,  0.0,  1.0,   1.0, 0.0, 0.0,
            -0.5,  0.5,  0.5,   0.0,  0.0,  1.0,   1.0, 0.0, 0.0,
            # Back face
             0.5, -0.5, -0.5,   0.0,  0.0, -1.0,   0.0, 1.0, 0.0,
            -0.5, -0.5, -0.5,   0.0,  0.0, -1.0,   0.0, 1.0, 0.0,
            -0.5,  0.5, -0.5,   0.0,  0.0, -1.0,   0.0, 1.0, 0.0,
             0.5,  0.5, -0.5,   0.0,  0.0, -1.0,   0.0, 1.0, 0.0,
            # Left face
            -0.5, -0.5, -0.5,  -1.0,  0.0,  0.0,   0.0, 0.0, 1.0,
            -0.5, -0.5,  0.5,  -1.0,  0.0,  0.0,   0.0, 0.0, 1.0,
            -0.5,  0.5,  0.5,  -1.0,  0.0,  0.0,   0.0, 0.0, 1.0,
            -0.5,  0.5, -0.5,  -1.0,  0.0,  0.0,   0.0, 0.0, 1.0,
            # Right face
             0.5, -0.5,  0.5,   1.0,  0.0,  0.0,   1.0, 1.0, 0.0,
             0.5, -0.5, -0.5,   1.0,  0.0,  0.0,   1.0, 1.0, 0.0,
             0.5,  0.5, -0.5,   1.0,  0.0,  0.0,   1.0, 1.0, 0.0,
             0.5,  0.5,  0.5,   1.0,  0.0,  0.0,   1.0, 1.0, 0.0,
            # Top face
            -0.5,  0.5,  0.5,   0.0,  1.0,  0.0,   0.0, 1.0, 1.0,
             0.5,  0.5,  0.5,   0.0,  1.0,  0.0,   0.0, 1.0, 1.0,
             0.5,  0.5, -0.5,   0.0,  1.0,  0.0,   0.0, 1.0, 1.0,
            -0.5,  0.5, -0.5,   0.0,  1.0,  0.0,   0.0, 1.0, 1.0,
            # Bottom face
            -0.5, -0.5, -0.5,   0.0, -1.0,  0.0,   1.0, 0.0, 1.0,
             0.5, -0.5, -0.5,   0.0, -1.0,  0.0,   1.0, 0.0, 1.0,
             0.5, -0.5,  0.5,   0.0, -1.0,  0.0,   1.0, 0.0, 1.0,
            -0.5, -0.5,  0.5,   0.0, -1.0,  0.0,   1.0, 0.0, 1.0,
        ]
        self.vbuf = self.engine.createBuffer(vertices, bz.BufferType.VERTEX)

        indices = []
        for i in range(6):
            indices.extend([i*4+0, i*4+1, i*4+2, i*4+2, i*4+3, i*4+0])
        
        self.ibuf = self.engine.createBuffer(indices, bz.BufferType.INDEX)

        ubo_size_bytes = (
            glm.sizeof(glm.mat4) * 2 +  
            glm.sizeof(glm.mat4) * 11 + 
            glm.sizeof(glm.vec4) * 3    
        )
        self.ubuf = self.engine.createBuffer(ubo_size_bytes, bz.BufferType.UNIFORM)

    def setup_descriptors(self):
        # Create Descriptor Pool and allocate descriptor set
        self.pool = self.engine.createDescriptorPool(max_sets=2, uniform_buffers=2)
        self.desc_set = self.pool.allocateFrameDescriptorSet(self.pipeline, set=0)
        self.desc_set.setBuffer(0, self.ubuf)

    def record_commands(self):
        # Create and record a command buffer
        self.cmd = self.engine.createCommandBuffer()
        self.cmd.begin()
        self.cmd.beginRendering(clear_color=[0.05, 0.05, 0.05, 1.0])
        self.cmd.setViewport()
        self.cmd.setScissor()
        self.cmd.bindPipeline(self.pipeline)
        self.cmd.bindDescriptorSet(self.desc_set, self.pipeline, set=0)
        self.cmd.bindVertexBuffer(self.vbuf)
        self.cmd.bindIndexBuffer(self.ibuf)
        self.cmd.drawIndexedInstanced(36, 11)
        self.cmd.endRendering()

    def on_update(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.frame_count += 1
        self.fps_timer += dt

        if self.fps_timer >= 1.0:
            avg_fps = self.frame_count / self.fps_timer
            self.engine.setTitle(f"Bazalt Demo - 10 Cubes Lighting | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            self.frame_count = 0
            self.fps_timer = 0.0

        mouse = self.engine.getMouseState()
        dx = mouse.dx - self.last_mouse_dx
        dy = mouse.dy - self.last_mouse_dy
        self.last_mouse_dx, self.last_mouse_dy = mouse.dx, mouse.dy

        right_vec = self.camera.update_mouse(dx, dy)
        self.camera.process_keyboard(self.engine, dt, right_vec)

        view, proj, _ = self.camera.get_matrices(1024.0 / 720.0)
        
        data = []
        
        for i in range(4):
            for j in range(4):
                data.append(view[i][j])
                
        for i in range(4):
            for j in range(4):
                data.append(proj[i][j])
                
        for i in range(10):
            angle = current_time * (0.5 + i * 0.1)
            radius = 5.0
            x = math.cos(i * math.pi * 2 / 10) * radius
            z = math.sin(i * math.pi * 2 / 10) * radius
            y = math.sin(current_time * 2.0 + i) * 1.5
            
            model = glm.translate(glm.mat4(1.0), glm.vec3(x, y, z))
            model = glm.rotate(model, angle, glm.vec3(1.0, 0.3, 0.5))
            
            for c in range(4):
                for r in range(4):
                    data.append(model[c][r])
                    
        light_pos_vec = glm.vec3(math.sin(current_time) * 10.0, 5.0, math.cos(current_time) * 10.0)
        
        light_model = glm.translate(glm.mat4(1.0), light_pos_vec)
        light_model = glm.scale(light_model, glm.vec3(0.5))
        for c in range(4):
            for r in range(4):
                data.append(light_model[c][r])

        data.extend([light_pos_vec.x, light_pos_vec.y, light_pos_vec.z, 0.0])
        data.extend([self.camera.pos.x, self.camera.pos.y, self.camera.pos.z, 0.0])
        data.extend([1.0, 0.9, 0.8, 0.0])
        
        self.ubuf.update(data)
        
        # Submit our pre-recorded command buffer to the GPU each frame
        self.engine.submit(self.cmd)

    def run(self):
        self.engine.run()

if __name__ == "__main__":
    app = App()
    app.run()
