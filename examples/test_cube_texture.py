import lumapy as lp
import glm
import time
import math

camera_pos = glm.vec3(0.0, 0.0, 3.0)
camera_front = glm.vec3(0.0, 0.0, -1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0)
yaw = -math.pi / 2.0
pitch = 0.0
last_mouse_dx = 0.0
last_mouse_dy = 0.0
last_time = time.time()

engine = lp.Engine()

@engine.onError
def error(msg):
    print(msg)

@engine.onFrame
def on_update():
    global camera_pos, camera_front, camera_up, yaw, pitch
    global last_mouse_dx, last_mouse_dy, last_time, ubuf

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    mouse = engine.getMouseState()
    sensitivity = 0.002
    
    dx = mouse.dx - last_mouse_dx
    dy = mouse.dy - last_mouse_dy
    last_mouse_dx = mouse.dx
    last_mouse_dy = mouse.dy

    yaw += dx * sensitivity
    pitch += dy * sensitivity
    pitch = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, pitch))

    front = glm.vec3(
        math.cos(yaw) * math.cos(pitch),
        math.sin(pitch),
        math.sin(yaw) * math.cos(pitch)
    )
    camera_front = glm.normalize(front)
    camera_right = glm.normalize(glm.cross(camera_front, glm.vec3(0.0, 1.0, 0.0)))
    camera_up = glm.normalize(glm.cross(camera_right, camera_front))

    speed = 2.5 * dt
    if engine.isKeyPressed(87): 
        camera_pos += speed * camera_front
    if engine.isKeyPressed(83): 
        camera_pos -= speed * camera_front
    if engine.isKeyPressed(65): 
        camera_pos -= speed * camera_right
    if engine.isKeyPressed(68): 
        camera_pos += speed * camera_right

    view = glm.lookAt(camera_pos, camera_pos + camera_front, camera_up)
    
    proj = glm.perspectiveRH_ZO(glm.radians(45.0), 1024.0 / 720.0, 0.1, 100.0)
    proj[1][1] *= -1 
    
    model = glm.mat4(1.0)
    mvp = proj * view * model
    
    ubuf.update(bytes(glm.transpose(mvp)))

    engine.submit(cmd)

if __name__ == "__main__":
    engine.init(1024, 720, "LumaPy Demo - Textured Multi-Cube")

    vert_spv = engine.compileShader("cube_tex.vert", lp.ShaderStage.VERTEX)
    frag_spv = engine.compileShader("cube_tex.frag", lp.ShaderStage.FRAGMENT)

    tex1 = engine.loadTexture("bricks.png")
    tex2 = engine.loadTexture("crate.jpg")

    pipeline = (engine.createPipeline()
        .vertexShader(vert_spv)
        .fragmentShader(frag_spv)
        .vertexFormat([lp.Format.FLOAT3, lp.Format.FLOAT3]) # pos, uv+texIndex
        .depthTest(True)
        .uniformBuffer(0, lp.ShaderStage.VERTEX)
        .texture(1, lp.ShaderStage.FRAGMENT)
        .texture(2, lp.ShaderStage.FRAGMENT)
        .build())

    # Format: pos x, y, z, uv u, v, texIndex
    # texIndex = 0.0 for bricks, 1.0 for crate
    vertices = [
        # Front face (Bricks)
        -0.5, -0.5,  0.5,   0.0, 0.0, 0.0,
         0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
         0.5,  0.5,  0.5,   1.0, 1.0, 0.0,
        -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
        # Back face (Bricks)
        -0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
         0.5, -0.5, -0.5,   0.0, 0.0, 0.0,
         0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
        -0.5,  0.5, -0.5,   1.0, 1.0, 0.0,
        # Left face (Crate)
        -0.5, -0.5, -0.5,   0.0, 0.0, 1.0,
        -0.5, -0.5,  0.5,   1.0, 0.0, 1.0,
        -0.5,  0.5,  0.5,   1.0, 1.0, 1.0,
        -0.5,  0.5, -0.5,   0.0, 1.0, 1.0,
        # Right face (Crate)
         0.5, -0.5, -0.5,   1.0, 0.0, 1.0,
         0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
         0.5,  0.5,  0.5,   0.0, 1.0, 1.0,
         0.5,  0.5, -0.5,   1.0, 1.0, 1.0,
        # Top face (Bricks)
        -0.5, -0.5, -0.5,   0.0, 1.0, 0.0,
         0.5, -0.5, -0.5,   1.0, 1.0, 0.0,
         0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
        -0.5, -0.5,  0.5,   0.0, 0.0, 0.0,
        # Bottom face (Crate)
        -0.5,  0.5, -0.5,   0.0, 0.0, 1.0,
         0.5,  0.5, -0.5,   1.0, 0.0, 1.0,
         0.5,  0.5,  0.5,   1.0, 1.0, 1.0,
        -0.5,  0.5,  0.5,   0.0, 1.0, 1.0,
    ]
    vbuf = engine.createBuffer(vertices, lp.BufferType.VERTEX, lp.DataType.FLOAT)

    indices = [
        0, 1, 2, 2, 3, 0,       # Front
        5, 4, 7, 7, 6, 5,       # Back
        8, 9, 10, 10, 11, 8,    # Left
        13, 12, 15, 15, 14, 13, # Right
        16, 17, 18, 18, 19, 16, # Top
        23, 22, 21, 21, 20, 23  # Bottom
    ]
    ibuf = engine.createBuffer(indices, lp.BufferType.INDEX, lp.DataType.UINT32)

    ubuf = engine.createBuffer([0.0]*16, lp.BufferType.UNIFORM, lp.DataType.FLOAT)

    cmd = engine.createCommandBuffer()
    
    cmd.begin()
    cmd.beginRendering(clear_color=[0.1, 0.2, 0.3, 1.0])
    cmd.setViewport()
    cmd.setScissor()
    cmd.bindPipeline(pipeline)
    cmd.bindUniformBuffer(0, ubuf, pipeline)
    cmd.bindTexture(1, tex1, pipeline)
    cmd.bindTexture(2, tex2, pipeline)
    cmd.bindVertexBuffer(vbuf)
    cmd.bindIndexBuffer(ibuf)
    cmd.drawIndexed(36)
    cmd.endRendering()

    engine.run()
