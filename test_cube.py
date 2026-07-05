import lumapy as lp
import glm
import time
import math
import collections

# Zmienne globalne stanu (zamiast użycia klasy App)
camera_pos = glm.vec3(0.0, 0.0, 3.0)
camera_front = glm.vec3(0.0, 0.0, -1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0)
yaw = -math.pi / 2.0
pitch = 0.0
last_mouse_dx = 0.0
last_mouse_dy = 0.0
last_time = time.time()
frame_times = collections.deque(maxlen=100)

engine = lp.Engine()

@engine.onError
def error(msg):
    print(f"[LumaPy Error]: {msg}")

@engine.onFrame
def on_update():
    global camera_pos, camera_front, camera_up, yaw, pitch
    global last_mouse_dx, last_mouse_dy, last_time, ubuf, frame_times

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    frame_times.append(dt)
    if len(frame_times) == 100:
        avg_dt = sum(frame_times) / 100.0
        avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0
        engine.setTitle(f"LumaPy Demo - 3D Cube | {dt*1000:.2f} ms/frame | {avg_fps:.1f} FPS")

    mouse = engine.getMouseState()
    sensitivity = 0.002
    
    dx = mouse.dx - last_mouse_dx
    dy = mouse.dy - last_mouse_dy
    last_mouse_dx = mouse.dx
    last_mouse_dy = mouse.dy

    yaw += dx * sensitivity
    # Zmieniono z - na +, aby myszka nie była "odwrócona" w osi Y.
    # Wartość dy rośnie, gdy przesuwamy myszkę w górę, więc '+' powoduje patrzenie w górę.
    pitch += dy * sensitivity

    # Clamp pitch to avoid flipping
    pitch = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, pitch))

    # Calculate forward, right, up vectors
    front = glm.vec3(
        math.cos(yaw) * math.cos(pitch),
        math.sin(pitch),
        math.sin(yaw) * math.cos(pitch)
    )
    camera_front = glm.normalize(front)
    camera_right = glm.normalize(glm.cross(camera_front, glm.vec3(0.0, 1.0, 0.0)))
    camera_up = glm.normalize(glm.cross(camera_right, camera_front))

    # Keyboard Input (W=87, A=65, S=83, D=68)
    speed = 2.5 * dt
    if engine.isKeyPressed(87): 
        camera_pos += speed * camera_front
    if engine.isKeyPressed(83): 
        camera_pos -= speed * camera_front
    if engine.isKeyPressed(65): 
        camera_pos -= speed * camera_right
    if engine.isKeyPressed(68): 
        camera_pos += speed * camera_right

    # Calculate Matrices
    view = glm.lookAt(camera_pos, camera_pos + camera_front, camera_up)
    
    # Zostawiamy perspectiveRH_ZO z odwróceniem osi Y, to tzw. standardowy hack pod Vulkana, 
    # używany by przenieść logikę z OpenGL na Vulkana (Vulkan ma Y w dół, zamiast w górę).
    proj = glm.perspectiveRH_ZO(glm.radians(45.0), 1024.0 / 720.0, 0.1, 100.0)
    proj[1][1] *= -1 
    
    model = glm.mat4(1.0)
    mvp = proj * view * model
    
    # Draw
    cmd.begin()
    cmd.beginRendering(clear_color=[0.1, 0.2, 0.3, 1.0])
    cmd.setViewport()
    cmd.setScissor()
    cmd.bindPipeline(pipeline)
    
    # Uaktualniamy nasz Uniform Buffer w każdej klatce (layout row-major/column-major tak jak wcześniej)
    ubuf.update(bytes(glm.transpose(mvp)))

    # Podpinamy zaktualizowany UBO pod binding = 0
    cmd.bindUniformBuffer(0, ubuf, pipeline)

    cmd.bindVertexBuffer(vbuf)
    cmd.bindIndexBuffer(ibuf)
    cmd.drawIndexed(36)
    cmd.endRendering()
    engine.submit(cmd)

if __name__ == "__main__":
    engine.init(1024, 720, "LumaPy Demo - 3D Cube")

    vert_spv = engine.compileShader("cube.vert", lp.ShaderStage.VERTEX)
    frag_spv = engine.compileShader("cube.frag", lp.ShaderStage.FRAGMENT)

    pipeline = (engine.createPipeline()
        .vertexShader(vert_spv)
        .fragmentShader(frag_spv)
        .vertexFormat([lp.Format.FLOAT3, lp.Format.FLOAT3])
        .depthTest(True)
        # .pushConstant(64, lp.ShaderStage.VERTEX) # Wykomentowane push constant
        .uniformBuffer(0, lp.ShaderStage.VERTEX)   # Dodany Uniform Buffer na binding=0
        .build())

    # Format: pos x, y, z, color r, g, b
    vertices = [
        # Front face
        -0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
         0.5, -0.5,  0.5,   0.0, 1.0, 0.0,
         0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
        -0.5,  0.5,  0.5,   1.0, 1.0, 0.0,
        # Back face
        -0.5, -0.5, -0.5,   1.0, 0.0, 1.0,
         0.5, -0.5, -0.5,   0.0, 1.0, 1.0,
         0.5,  0.5, -0.5,   1.0, 1.0, 1.0,
        -0.5,  0.5, -0.5,   0.0, 0.0, 0.0,
    ]
    vbuf = engine.createBuffer(vertices, lp.BufferType.VERTEX, lp.DataType.FLOAT)

    indices = [
        # Front
        0, 1, 2, 2, 3, 0,
        # Back
        5, 4, 7, 7, 6, 5,
        # Left
        4, 0, 3, 3, 7, 4,
        # Right
        1, 5, 6, 6, 2, 1,
        # Top
        3, 2, 6, 6, 7, 3,
        # Bottom
        4, 5, 1, 1, 0, 4
    ]
    ibuf = engine.createBuffer(indices, lp.BufferType.INDEX, lp.DataType.UINT32)

    # Inicjujemy Uniform Buffer (16 floatów, czyli 64 bajty na mat4)
    ubuf = engine.createBuffer([0.0]*16, lp.BufferType.UNIFORM, lp.DataType.FLOAT)

    cmd = engine.createCommandBuffer()

    # Wystartuj pętle głowną, domyślnie none jeśli bez klasy.
    engine.run()
