import lumapy as lp
import glm
import time
import math
import struct

camera_pos = glm.vec3(0.0, 0.0, 5.0)
camera_front = glm.vec3(0.0, 0.0, -1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0)
yaw = -math.pi / 2.0
pitch = 0.0
last_mouse_dx = 0.0
last_mouse_dy = 0.0

last_time = time.time()
frame_count = 0
fps_timer = 0.0

engine = lp.Engine()

@engine.onError
def error(msg):
    print(msg)

@engine.onFrame
def on_update():
    global camera_pos, camera_front, camera_up, yaw, pitch
    global last_mouse_dx, last_mouse_dy, last_time, ubuf
    global frame_count, fps_timer

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    frame_count += 1
    fps_timer += dt

    if fps_timer >= 1.0:
        avg_fps = frame_count / fps_timer
        engine.setTitle(f"LumaPy Demo - 10 Cubes Lighting | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
        frame_count = 0
        fps_timer = 0.0

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

    speed = 5.0 * dt
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
    
    # We pack the layout(binding = 0) uniform UBO manually
    data = []
    
    # view matrix (16 floats) - PyGLM m[i][j] gives column i, row j. 
    # Appending m[i][j] sequentially gives column-major order, which Vulkan expects.
    for i in range(4):
        for j in range(4):
            data.append(view[i][j])
            
    # proj matrix (16 floats)
    for i in range(4):
        for j in range(4):
            data.append(proj[i][j])
            
    # 10 models (160 floats)
    for i in range(10):
        # Place cubes in a circle
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
                
    # lightPos calculation
    light_pos_vec = glm.vec3(math.sin(current_time) * 10.0, 5.0, math.cos(current_time) * 10.0)
    
    # 11th model is the light source (scale it down so it's a small point)
    light_model = glm.translate(glm.mat4(1.0), light_pos_vec)
    light_model = glm.scale(light_model, glm.vec3(0.5)) # make it a smaller cube
    for c in range(4):
        for r in range(4):
            data.append(light_model[c][r])

    # lightPos (3 floats + 1 pad)
    data.extend([light_pos_vec.x, light_pos_vec.y, light_pos_vec.z, 0.0])
    
    # viewPos (3 floats + 1 pad)
    data.extend([camera_pos.x, camera_pos.y, camera_pos.z, 0.0])
    
    # lightColor (3 floats + 1 pad)
    data.extend([1.0, 0.9, 0.8, 0.0])
    
    # Update directly with list, API now handles packing!
    ubuf.update(data)

    engine.submit(cmd)

if __name__ == "__main__":
    engine.init(1024, 720, "LumaPy Demo - 10 Cubes Lighting")

    vert_spv = engine.compileShader("10cubes.vert", lp.ShaderStage.VERTEX)
    frag_spv = engine.compileShader("10cubes.frag", lp.ShaderStage.FRAGMENT)

    pipeline = (engine.createPipeline()
        .vertexShader(vert_spv)
        .fragmentShader(frag_spv)
        .vertexFormat([lp.Format.FLOAT3, lp.Format.FLOAT3, lp.Format.FLOAT3]) # pos, normal, color
        .depthTest(True)
        .uniformBuffer(0, lp.ShaderStage.VERTEX)   
        .uniformBuffer(0, lp.ShaderStage.FRAGMENT) 
        .build())

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
    # No more DataType.FLOAT required, inferred automatically!
    vbuf = engine.createBuffer(vertices, lp.BufferType.VERTEX)

    indices = []
    for i in range(6):
        indices.extend([i*4+0, i*4+1, i*4+2, i*4+2, i*4+3, i*4+0])
    
    # Automatically inferred as UINT32 for indices
    ibuf = engine.createBuffer(indices, lp.BufferType.INDEX)

    # Calculate UBO size dynamically using glm.sizeof to avoid hardcoding sizes.
    # std140 layout aligns vec3 to 16 bytes (so we treat them as vec4 for sizing)
    ubo_size_bytes = (
        glm.sizeof(glm.mat4) * 2 +  # view, proj
        glm.sizeof(glm.mat4) * 11 + # 10 models + 1 light model
        glm.sizeof(glm.vec4) * 3    # lightPos, viewPos, lightColor
    )
    ubuf = engine.createBuffer(ubo_size_bytes, lp.BufferType.UNIFORM)

    cmd = engine.createCommandBuffer()
    
    cmd.begin()
    cmd.beginRendering(clear_color=[0.05, 0.05, 0.05, 1.0])
    cmd.setViewport()
    cmd.setScissor()
    cmd.bindPipeline(pipeline)
    cmd.bindUniformBuffer(0, ubuf, pipeline)
    cmd.bindVertexBuffer(vbuf)
    cmd.bindIndexBuffer(ibuf)
    # Draw 36 indices, with 11 instances (10 cubes + 1 light source)
    cmd.drawIndexedInstanced(36, 11)
    cmd.endRendering()

    engine.run()
