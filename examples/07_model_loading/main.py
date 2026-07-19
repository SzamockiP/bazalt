import os
import time
import math
import numpy as np
import trimesh
import glm
import bazalt as bz

class Camera:
    def __init__(self, pos=(0.0, 2.0, 10.0), yaw=-math.pi/2):
        self.pos = glm.vec3(*pos)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = yaw
        self.pitch = 0.0
        self.sensitivity = 0.002
        self.speed = 10.0

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
        proj = glm.perspectiveRH_ZO(glm.radians(60.0), aspect_ratio, 0.1, 1000.0)
        proj[1][1] *= -1 
        model = glm.mat4(1.0)
        return view, proj, model


def load_materials(mtl_path):
    textures = {}
    if not os.path.exists(mtl_path): 
        return textures
        
    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        current_mat = None
        for line in f:
            line = line.strip()
            if line.startswith('newmtl '):
                current_mat = line.split(' ', 1)[1].strip()
                textures[current_mat] = {'texture': None, 'color': [1.0, 1.0, 1.0]}
            elif line.startswith('Kd ') and current_mat:
                parts = line.split()[1:]
                textures[current_mat]['color'] = [float(parts[0]), float(parts[1]), float(parts[2])]
            elif line.startswith('map_Kd ') and current_mat:
                tex_file = line.split(' ', 1)[1].strip().replace('\\', '/')
                textures[current_mat]['texture'] = tex_file
    return textures


class DemoApp:
    def __init__(self):
        # Create window, logger and renderer
        self.logger = bz.Logger()
        self.logger.on_message(self.on_message)
        self.window = bz.Window(1024, 720, "Bazalt Demo - Model Loader", logger=self.logger)
        self.ctx = bz.Context(self.logger)
        self.renderer = bz.SwapchainRenderer(self.window, self.ctx)
        self.window.set_cursor_mode(bz.CURSOR_DISABLED)
        
        self.camera = Camera()
        self.last_time = time.time()
        self.last_mouse_dx = 0.0
        self.last_mouse_dy = 0.0
        self.frame_count = 0
        self.fps_timer = 0.0
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.normpath(os.path.join(script_dir, "..", "assets"))
        
        self.setup_pipeline(script_dir)
        self.load_scene(os.path.join(self.assets_dir, "San_Miguel", "san-miguel.obj"))
        self.setup_descriptors()
        self.record_commands()

    def on_message(self, msg):
        print(f"[{msg.severity}] {msg.text}")

    def setup_pipeline(self, script_dir):
        # Compile GLSL shaders
        vert_spv = self.ctx.compile_shader(os.path.join(script_dir, "model.vert"), bz.ShaderStage.VERTEX)
        frag_spv = self.ctx.compile_shader(os.path.join(script_dir, "model.frag"), bz.ShaderStage.FRAGMENT)

        # Create a graphics pipeline using a builder pattern
        self.pipeline = (self.ctx.graphics_pipeline()
            .vertex_shader(vert_spv)
            .fragment_shader(frag_spv)
            .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT3])
            .depth_test(True)
            .cull_mode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE)
            .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
            .texture(0, bz.ShaderStage.FRAGMENT, set=1)
            .build(self.renderer))

        # 3 * mat4 = 192 bytes
        self.ubuf = self.ctx.create_buffer(192, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)

    def load_scene(self, obj_path):
        print("Loading model...")
        obj_dir = os.path.dirname(obj_path) or "."
        mtl_textures = load_materials(obj_path.replace('.obj', '.mtl'))
        
        scene = trimesh.load(obj_path, process=False)
        self.loaded_textures = {}
        
        white_png_path = os.path.join(self.assets_dir, "white.png")
        self.default_texture = self.ctx.load_image(white_png_path)
        
        all_vertices, all_normals, all_uvs, all_colors, all_faces = [], [], [], [], []
        self.draw_calls = []
        vertex_offset, index_offset = 0, 0
        
        for name, geom in scene.geometry.items():
            vertices = geom.vertices
            faces = geom.faces
            
            if not hasattr(geom, 'vertex_normals') or geom.vertex_normals is None or len(geom.vertex_normals) != len(vertices):
                geom.fix_normals()
                
            normals = geom.vertex_normals
            if normals is None or len(normals) != len(vertices):
                normals = np.zeros_like(vertices)
                normals[:, 1] = 1.0
            
            if hasattr(geom.visual, 'uv') and geom.visual.uv is not None and len(geom.visual.uv) == len(vertices):
                uvs = geom.visual.uv
            else:
                uvs = np.zeros((len(vertices), 2), dtype=np.float32)
                
            mat_color = [1.0, 1.0, 1.0]
            tex = self.default_texture
            if hasattr(geom.visual, 'material') and hasattr(geom.visual.material, 'name'):
                mat_name = geom.visual.material.name
                mat_info = mtl_textures.get(mat_name)
                
                if mat_info:
                    mat_color = mat_info['color']
                    tex_file = mat_info['texture']
                    
                    if tex_file:
                        tex_file = tex_file if os.path.isabs(tex_file) else os.path.normpath(os.path.join(obj_dir, tex_file))
                        if os.path.exists(tex_file):
                            if tex_file not in self.loaded_textures:
                                self.loaded_textures[tex_file] = self.ctx.load_image(tex_file)
                            tex = self.loaded_textures[tex_file]
            
            colors = np.tile(mat_color, (len(vertices), 1)).astype(np.float32)
            
            all_vertices.append(vertices)
            all_normals.append(normals)
            all_uvs.append(uvs)
            all_colors.append(colors)
            all_faces.append(faces)
                        
            self.draw_calls.append({
                'index_count': len(faces) * 3,
                'vertex_offset': vertex_offset,
                'first_index': index_offset,
                'texture': tex
            })
            
            vertex_offset += len(vertices)
            index_offset += len(faces) * 3

        print(f"Model loaded. Submeshes: {len(self.draw_calls)}, textures: {len(self.loaded_textures)}")

        interleaved = np.empty((vertex_offset, 11), dtype=np.float32)
        interleaved[:, 0:3] = np.concatenate(all_vertices)
        interleaved[:, 3:6] = np.concatenate(all_normals)
        interleaved[:, 6:8] = np.concatenate(all_uvs)
        interleaved[:, 8:11] = np.concatenate(all_colors)
        
        self.vbuf = self.ctx.create_buffer(interleaved.flatten(), bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
        self.ibuf = self.ctx.create_buffer(np.concatenate(all_faces).flatten().astype(np.uint32), bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

    def setup_descriptors(self):
        # Create Descriptor Pool and allocate descriptor set
        self.pool = self.ctx.create_descriptor_pool(
            max_sets=3 + len(self.loaded_textures), 
            uniform_buffers=2, 
            samplers=1 + len(self.loaded_textures)
        )
        self.frame_set = self.pool.allocate_frame_set(self.pipeline, set=0)
        self.frame_set.set_buffer(0, self.ubuf)

        self.texture_sets = {}
        for path, tex in self.loaded_textures.items():
            tex_set = self.pool.allocate_set(self.pipeline, set=1)
            tex_set.set_image(0, tex)
            self.texture_sets[tex] = tex_set
            
        default_tex_set = self.pool.allocate_set(self.pipeline, set=1)
        default_tex_set.set_image(0, self.default_texture)
        self.texture_sets[self.default_texture] = default_tex_set

    def record_commands(self):
        # Create and record a command buffer
        self.cmd = self.ctx.create_command_buffer()
        self.cmd.begin()
        self.cmd.begin_rendering(self.renderer, clear_color=[0.1, 0.2, 0.3, 1.0])
        self.cmd.bind_pipeline(self.pipeline)
        self.cmd.bind_descriptor_set(self.frame_set, self.pipeline, set=0)
        
        self.cmd.bind_vertex_buffer(self.vbuf)
        self.cmd.bind_index_buffer(self.ibuf)
        
        for dc in self.draw_calls:
            self.cmd.bind_descriptor_set(self.texture_sets[dc['texture']], self.pipeline, set=1)
            self.cmd.draw_indexed(dc['index_count'], first_index=dc['first_index'], vertex_offset=dc['vertex_offset'])
            
        self.cmd.end_rendering(self.renderer)

    def run(self):
        print("Rendering started")
        last_mouse_dx = self.last_mouse_dx
        last_mouse_dy = self.last_mouse_dy
        
        while self.window.is_open():
            self.window.poll_events()
            
            if frame := self.renderer.begin_frame():
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time
                
                self.frame_count += 1
                self.fps_timer += dt
                if self.fps_timer >= 1.0:
                    avg_fps = self.frame_count / self.fps_timer
                    self.window.set_title(f"Bazalt Demo - Model Loader | {1000.0/avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
                    self.frame_count = 0
                    self.fps_timer = 0.0
                    
                mouse = self.window.get_mouse_state()
                dx = mouse.dx - last_mouse_dx
                dy = mouse.dy - last_mouse_dy
                last_mouse_dx, last_mouse_dy = mouse.dx, mouse.dy
                
                right_vec = self.camera.update_mouse(dx, dy)
                self.camera.process_keyboard(self.window, dt, right_vec)
                
                view, proj, model = self.camera.get_matrices(1024.0 / 720.0)
                self.ubuf.update(view.to_bytes() + proj.to_bytes() + model.to_bytes())
                
                frame.submit(self.cmd)

if __name__ == "__main__":
    app = DemoApp()
    app.run()
