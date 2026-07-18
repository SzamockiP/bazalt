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

    def process_keyboard(self, window, dt, right):
        velocity = self.speed * dt
        if window.is_key_pressed(bz.KEY_W): self.pos += velocity * self.front
        if window.is_key_pressed(bz.KEY_S): self.pos -= velocity * self.front
        if window.is_key_pressed(bz.KEY_A): self.pos -= velocity * right
        if window.is_key_pressed(bz.KEY_D): self.pos += velocity * right
        if window.is_key_pressed(bz.KEY_SPACE): self.pos += velocity * self.up
        if window.is_key_pressed(bz.KEY_LEFT_SHIFT): self.pos -= velocity * self.up

    def get_matrices(self, aspect_ratio):
        view = glm.lookAt(self.pos, self.pos + self.front, self.up)
        proj = glm.perspectiveRH_ZO(glm.radians(45.0), aspect_ratio, 0.1, 100.0)
        proj[1][1] *= -1 
        model = glm.mat4(1.0)
        return view, proj, model

# Create window, logger, and renderer
logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - 10 Cubes Lighting")
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)
window.set_cursor_mode(bz.CURSOR_DISABLED)

# Compile shaders
vert_spv = ctx.compile_shader("10cubes.vert", bz.ShaderStage.VERTEX)
frag_spv = ctx.compile_shader("10cubes.frag", bz.ShaderStage.FRAGMENT)

# Build pipeline
pipeline = (ctx.pipeline_builder()
    .vertex_shader(vert_spv)
    .fragment_shader(frag_spv)
    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3]) # pos, normal, color
    .depth_test(True)
    .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)   
    .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0) 
    .build(renderer))

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
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)

indices = []
for i in range(6):
    indices.extend([i*4+0, i*4+1, i*4+2, i*4+2, i*4+3, i*4+0])

ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

ubo_size_bytes = (
    glm.sizeof(glm.mat4) * 2 +  
    glm.sizeof(glm.mat4) * 11 + 
    glm.sizeof(glm.vec4) * 3    
)
ubuf = ctx.create_buffer(ubo_size_bytes, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)

# Descriptors
pool = ctx.create_descriptor_pool(max_sets=2, uniform_buffers=2)
desc_set = pool.allocate_frame_set(pipeline, set=0)
desc_set.set_buffer(0, ubuf)

# Record commands
cmd = ctx.create_command_buffer()
cmd.begin()
cmd.begin_rendering(renderer, clear_color=[0.05, 0.05, 0.05, 1.0])
cmd.bind_pipeline(pipeline)
cmd.bind_descriptor_set(desc_set, pipeline, set=0)
cmd.bind_vertex_buffer(vbuf)
cmd.bind_index_buffer(ibuf)
cmd.draw_indexed_instanced(36, 11)
cmd.end_rendering(renderer)

# Main loop
camera = Camera(pos=(0.0, 0.0, 5.0), speed=5.0)
last_mouse_dx = 0.0
last_mouse_dy = 0.0
last_time = time.time()
frame_count = 0
fps_timer = 0.0

while window.is_open():
    window.poll_events()

    if frame := renderer.begin_frame():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        frame_count += 1
        fps_timer += dt

        if fps_timer >= 1.0:
            avg_fps = frame_count / fps_timer
            window.set_title(f"Bazalt Demo - 10 Cubes Lighting | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            frame_count = 0
            fps_timer = 0.0

        mouse = window.get_mouse_state()
        dx = mouse.dx - last_mouse_dx
        dy = mouse.dy - last_mouse_dy
        last_mouse_dx, last_mouse_dy = mouse.dx, mouse.dy

        right_vec = camera.update_mouse(dx, dy)
        camera.process_keyboard(window, dt, right_vec)

        view, proj, _ = camera.get_matrices(1024.0 / 720.0)
        
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
        data.extend([camera.pos.x, camera.pos.y, camera.pos.z, 0.0])
        data.extend([1.0, 0.9, 0.8, 0.0])
        
        ubuf.update(data)
        
        frame.submit(cmd)
