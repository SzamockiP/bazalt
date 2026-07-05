import lumapy as lp
import time

engine = lp.Engine()

@engine.onError
def error(msg):
    print(msg)

last_time = time.time()
frame_count = 0
fps_timer = 0.0

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
        engine.setTitle(f"LumaPy Demo - Empty | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
        frame_count = 0
        fps_timer = 0.0

    engine.submit(cmd)

if __name__ == "__main__":
    # Disable VSync in engine initialization if supported (assuming standard config, or if we can)
    # The current engine.init just takes width, height, title. VSync might be hardcoded in Swapchain.
    engine.init(1024, 720, "LumaPy Demo - Empty")

    cmd = engine.createCommandBuffer()
    
    # We only record the command buffer once!
    cmd.begin()
    cmd.beginRendering(clear_color=[0.1, 0.1, 0.1, 1.0])
    cmd.endRendering()

    engine.run()
