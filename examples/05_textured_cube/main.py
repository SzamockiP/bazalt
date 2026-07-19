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

window = bz.Window(1024, 720, "Bazalt Demo - Textured Multi-Cube")
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)
window.set_cursor_mode(bz.CURSOR_DISABLED)

# Compile shaders
vert_spv = ctx.compile_shader("cube_tex.vert", bz.ShaderStage.VERTEX)
frag_spv = ctx.compile_shader("cube_tex.frag", bz.ShaderStage.FRAGMENT)

# Build pipeline
pipeline = (ctx.graphics_pipeline()
    .vertex_shader(vert_spv)
    .fragment_shader(frag_spv)
    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT2])
    .depth_test(True)
    .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
    .texture(0, bz.ShaderStage.FRAGMENT, set=1)
    .build(renderer))

# Vertex data: pos (x,y,z), uv (u,v)
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
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)

indices = [
    # --- BRICKS ---
    0, 1, 2, 2, 3, 0,       # Front
    5, 4, 7, 7, 6, 5,       # Back
    16, 17, 18, 18, 19, 16, # Top
    # --- CRATE ---
    8, 9, 10, 10, 11, 8,    # Left
    13, 12, 15, 15, 14, 13, # Right
    23, 22, 21, 21, 20, 23  # Bottom
]
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC, bz.DataType.UINT32)

ubuf = ctx.create_buffer([0.0]*16, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC, bz.DataType.FLOAT)

# Load textures and create descriptors
tex1 = ctx.load_image("../assets/wall.png")
tex2 = ctx.load_image("../assets/container.png")

pool = ctx.create_descriptor_pool(max_sets=4, samplers=2, uniform_buffers=2)

frame_set = pool.allocate_frame_set(pipeline, set=0)
frame_set.set_buffer(0, ubuf)

bricks_set = pool.allocate_set(pipeline, set=1)
bricks_set.set_image(0, tex1)

crate_set = pool.allocate_set(pipeline, set=1)
crate_set.set_image(0, tex2)

# Record commands
cmd = ctx.create_command_buffer()
cmd.begin()
cmd.begin_rendering(renderer, clear_color=[0.1, 0.2, 0.3, 1.0])
cmd.bind_pipeline(pipeline)
cmd.bind_descriptor_set(frame_set, pipeline, set=0)
cmd.bind_vertex_buffer(vbuf)
cmd.bind_index_buffer(ibuf)

# Bricks (18 indices)
cmd.bind_descriptor_set(bricks_set, pipeline, set=1)
cmd.draw_indexed(18, first_index=0)

# Crate (18 indices)
cmd.bind_descriptor_set(crate_set, pipeline, set=1)
cmd.draw_indexed(18, first_index=18)

cmd.end_rendering(renderer)

# Main loop
camera = Camera(pos=(0.0, 0.0, 3.0), speed=2.5)
last_time = time.time()
last_mouse_dx = 0.0
last_mouse_dy = 0.0
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
            window.set_title(f"Bazalt Demo - Textured Multi-Cube | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            frame_count = 0
            fps_timer = 0.0

        mouse = window.get_mouse_state()
        dx = mouse.dx - last_mouse_dx
        dy = mouse.dy - last_mouse_dy
        last_mouse_dx, last_mouse_dy = mouse.dx, mouse.dy

        right_vec = camera.update_mouse(dx, dy)
        camera.process_keyboard(window, dt, right_vec)

        view, proj, model = camera.get_matrices(1024.0 / 720.0)
        mvp = proj * view * model
        
        ubuf.update(bytes(glm.transpose(mvp)))
        
        frame.submit(cmd)
