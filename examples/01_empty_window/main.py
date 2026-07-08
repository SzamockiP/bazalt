import bazalt as bz
import time

engine = bz.Engine()

# Optional: Register an error callback
@engine.onError
def error(msg):
    print(msg)

last_time = time.time()
frame_count = 0
fps_timer = 0.0

# Register a per-frame callback
@engine.onFrame
def on_update():
    global last_time, frame_count, fps_timer

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time
    
    frame_count += 1
    fps_timer += dt
    
    # Update window title once per second to avoid OS overhead of setting title every frame
    if fps_timer >= 1.0:
        avg_fps = frame_count / fps_timer
        engine.setTitle(f"Bazalt Demo - Empty | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
        frame_count = 0
        fps_timer = 0.0

    # Submit our pre-recorded command buffer to the GPU each frame
    engine.submit(cmd)

if __name__ == "__main__":
    engine.init(1024, 720, "Bazalt Demo - Empty")

    cmd = engine.createCommandBuffer()
    
    # We only record the command buffer once!
    cmd.begin()
    # Start rendering pass with a dark gray clear color
    cmd.beginRendering(clear_color=[0.1, 0.1, 0.1, 1.0])
    cmd.endRendering()

    engine.run()
