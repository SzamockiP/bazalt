import bazalt as bz

engine = bz.Engine()

# Optional: Register an error callback
@engine.onError
def error(msg):
    print(msg)

# Register a per-frame callback
@engine.onFrame
def on_update():
    # Submit our pre-recorded command buffer to the GPU each frame
    engine.submit(cmd)

if __name__ == "__main__":
    engine.init(800, 600, "Bazalt Demo - Textured Quad")

    vert_spv = engine.compileShader("quad_tex.vert", bz.ShaderStage.VERTEX)
    frag_spv = engine.compileShader("quad_tex.frag", bz.ShaderStage.FRAGMENT)

    # Load a texture from file
    texture = engine.loadTexture("../assets/wall.png")

    # The pipeline defines how geometry is drawn. We explicitly map our vertex layout
    # and specify that our fragment shader expects a texture bound to slot 0.
    pipeline = (engine.createPipeline()
        .vertexShader(vert_spv)
        .fragmentShader(frag_spv)
        .vertexFormat([bz.Format.FLOAT2, bz.Format.FLOAT2]) # Position + UV
        .texture(0, bz.ShaderStage.FRAGMENT, set=0) # Bind texture to slot 0
        .build())

    # Geometry with interleaved Position (x,y) and UV (u,v)
    vertices = [
        -0.5, -0.5,  0.0, 0.0,
         0.5, -0.5,  1.0, 0.0,
         0.5,  0.5,  1.0, 1.0,
        -0.5,  0.5,  0.0, 1.0,
    ]
    # Create Vertex Buffer
    vbuf = engine.createBuffer(vertices, bz.BufferType.VERTEX, bz.DataType.FLOAT)

    indices = [
        0, 1, 2, 2, 3, 0
    ]
    # Create Index Buffer
    ibuf = engine.createBuffer(indices, bz.BufferType.INDEX, bz.DataType.UINT32)

    # Descriptors map GPU resources (like textures) to shader bindings.
    # We allocate a descriptor set and update it with our loaded texture.
    pool = engine.createDescriptorPool(max_sets=1, samplers=1)
    desc_set = pool.allocateDescriptorSet(pipeline, set=0)
    
    # Bind the texture to the descriptor set
    desc_set.setTexture(0, texture)

    cmd = engine.createCommandBuffer()
    
    cmd.begin()
    cmd.beginRendering(clear_color=[0.1, 0.2, 0.3, 1.0])
    cmd.setViewport()
    cmd.setScissor()
    # Bind resources
    cmd.bindPipeline(pipeline)
    cmd.bindDescriptorSet(desc_set, pipeline, set=0)
    cmd.bindVertexBuffer(vbuf)
    cmd.bindIndexBuffer(ibuf)
    
    # Draw 6 indices (2 triangles = 1 quad)
    cmd.drawIndexed(6)
    cmd.endRendering()

    engine.run()
