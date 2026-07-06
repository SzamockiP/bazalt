import bazalt as bz

engine = bz.Engine()

@engine.onError
def error(msg):
    print(msg)

@engine.onFrame
def on_update():
    engine.submit(cmd)

if __name__ == "__main__":
    engine.init(800, 600, "Bazalt Demo - Textured Quad")

    vert_spv = engine.compileShader("quad_tex.vert", bz.ShaderStage.VERTEX)
    frag_spv = engine.compileShader("quad_tex.frag", bz.ShaderStage.FRAGMENT)

    # Load texture
    texture = engine.loadTexture("bricks.png")

    pipeline = (engine.createPipeline()
        .vertexShader(vert_spv)
        .fragmentShader(frag_spv)
        .vertexFormat([bz.Format.FLOAT2, bz.Format.FLOAT2])
        .texture(0, bz.ShaderStage.FRAGMENT)
        .build())

    # Format: pos x, y, uv u, v
    vertices = [
        -0.5, -0.5,  0.0, 0.0,
         0.5, -0.5,  1.0, 0.0,
         0.5,  0.5,  1.0, 1.0,
        -0.5,  0.5,  0.0, 1.0,
    ]
    vbuf = engine.createBuffer(vertices, bz.BufferType.VERTEX, bz.DataType.FLOAT)

    indices = [
        0, 1, 2, 2, 3, 0
    ]
    ibuf = engine.createBuffer(indices, bz.BufferType.INDEX, bz.DataType.UINT32)

    cmd = engine.createCommandBuffer()
    
    cmd.begin()
    cmd.beginRendering(clear_color=[0.1, 0.2, 0.3, 1.0])
    cmd.setViewport()
    cmd.setScissor()
    cmd.bindPipeline(pipeline)
    cmd.bindTexture(0, texture, pipeline)
    cmd.bindVertexBuffer(vbuf)
    cmd.bindIndexBuffer(ibuf)
    cmd.drawIndexed(6)
    cmd.endRendering()

    engine.run()
