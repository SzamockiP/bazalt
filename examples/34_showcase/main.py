"""Showcase — San Miguel through most of the engine at once.

One scene exercises the features that examples 09-32 introduce one at a time:

  * GPU frustum culling per submesh (compute) into a static multi-draw
    indirect buffer — the CPU never learns what survived. Unlike example 28
    there is no compaction: each submesh owns a pre-filled command and the
    compute pass only flips its instanceCount between 0 and 1.
  * One bindless array holds every material. The indirect command's
    firstInstance carries the material index, so the shader reads it back as
    gl_InstanceIndex — no gl_DrawID, no per-draw descriptor binds.
  * A day-night cycle moves the sun. At night the moon takes over the light
    and the shadow map. The sky is procedural: gradient, drifting fbm clouds
    and the sun disc by day, stars and a moon by night.
  * PCF shadows from one 4096 depth-only pass with depth_bias() and alpha
    discard, so leaves cast leaf-shaped shadows. The light's ortho FOLLOWS
    the camera (a 22 m window, texel-snapped) instead of covering the scene —
    one map keeps the texel density cascades exist for. The receiver samples
    a rotated Poisson disk over the hardware PCF.
  * Glass renders in a second blended pass over the opaques. The .mtl's
    dissolve/illum-4 transparency lands in the material pixel's alpha, the
    glass commands leave the opaque indirect buffer (their indexCount is 0
    there), and the shadow shader's discard drops them.
  * Normal mapping from the .mtl's map_Bump entries through a second bindless
    array. The tangent frame comes from screen-space derivatives, so no
    vertex attribute carries it. Grayscale bump maps turn into normal maps
    through a height gradient at load.
  * The scene renders into an HDR (RGBA16F) target with MSAA and
    alpha-to-coverage. MSAA refuses a preserving second pass, so sky, scene
    and fireflies are three pipelines inside ONE rendering scope.
  * A post chain adds bloom, god rays, a lens flare, ACES tone mapping, a
    time-of-day grade and a vignette. The god-ray mask falls out of the depth
    buffer: sky pixels keep the cleared depth.
  * Fireflies fly at night. A compute pass integrates them, a point draw
    renders them, and the scene shader reads the same buffer as a list of
    small point lights.
  * Profiling is first class: cmd.label() around every pass, name= on every
    resource, cmd.timer() per pass and gpu_timing for the frame total. Open a
    capture in Nsight or RenderDoc and the frame reads like this docstring.

The first start parses the 1.1 GB OBJ with trimesh (minutes) and caches the
result to a .npz next to it. Later starts skip trimesh and load in seconds.
Get the model with examples/assets/download_san_miguel.py. Expect roughly
1.5-2 GB of VRAM: ~300 mipmapped textures plus the HDR MSAA targets.

Deliberate shortcuts: the internal resolution is fixed at the start-up window
size (resizing only stretches the composite), leaf opacity comes from the
diffuse texture's alpha (map_d is ignored), fireflies cast no shadows,
geometry more than ~22 m from the camera renders unshadowed (the map's white
border), and the glass pass draws unsorted.

Requires Feature.BINDLESS and Feature.MULTI_DRAW_INDIRECT; the demo exits
with a message without them. Hot reload is on — edit any shader (sky.frag,
composite.frag) while it runs.

    WASD + mouse   fly (SPACE up, LEFT SHIFT down)
    P              pause the time of day
    LEFT / RIGHT   scrub the time of day
    UP / DOWN      speed the cycle up / down
    F              cycle windowed -> frameless -> fullscreen -> borderless
    ESC            quit
"""

import math
import os
import struct
import time

import glm
import numpy as np

import bazalt as bz

W, H = 1600, 900             # internal resolution, fixed for the run
SHADOW_SIZE = 4096
SHADOW_RADIUS = 22.0         # the shadow map covers this box around the camera
FIREFLY_COUNT = 128
DAY_SECONDS = 240.0          # one full day-night cycle at speed 1
SUN_AZIMUTH = 0.55           # radians; fixed heading of the sun's arc
CACHE_VERSION = 4            # v4: flipped V, per-slot alpha (glass)

WINDOW_MODES = [bz.WindowMode.WINDOWED, bz.WindowMode.FRAMELESS,
                bz.WindowMode.FULLSCREEN, bz.WindowMode.FULLSCREEN_WINDOWED]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Camera:
    """The FPS camera from example 07."""

    def __init__(self, pos=(0.0, 2.0, 10.0), yaw=-math.pi / 2):
        self.pos = glm.vec3(*pos)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = yaw
        self.pitch = 0.0
        self.sensitivity = 0.002
        self.speed = 8.0

    def update_mouse(self, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch = clamp(self.pitch + dy * self.sensitivity,
                           -math.pi / 2 + 0.01, math.pi / 2 - 0.01)
        self.front = glm.normalize(glm.vec3(
            math.cos(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            math.sin(self.yaw) * math.cos(self.pitch)))
        right = glm.normalize(glm.cross(self.front, glm.vec3(0.0, 1.0, 0.0)))
        self.up = glm.normalize(glm.cross(right, self.front))
        return right

    def process_keyboard(self, window, dt, right):
        v = self.speed * dt
        if window.is_key_pressed(bz.KEY_W): self.pos += v * self.front
        if window.is_key_pressed(bz.KEY_S): self.pos -= v * self.front
        if window.is_key_pressed(bz.KEY_A): self.pos -= v * right
        if window.is_key_pressed(bz.KEY_D): self.pos += v * right
        if window.is_key_pressed(bz.KEY_SPACE): self.pos += v * self.up
        if window.is_key_pressed(bz.KEY_LEFT_SHIFT): self.pos -= v * self.up

    def view_proj(self):
        view = glm.lookAt(self.pos, self.pos + self.front, self.up)
        proj = glm.perspectiveRH_ZO(glm.radians(60.0), W / H, 0.1, 500.0)
        proj[1][1] *= -1
        return proj * view


def load_materials(mtl_path):
    """Kd / map_Kd / map_Bump / transparency per material (07's parser grown)."""
    materials = {}
    if not os.path.exists(mtl_path):
        return materials
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        current = None
        for line in f:
            line = line.strip()
            if line.startswith("newmtl "):
                current = line.split(" ", 1)[1].strip()
                materials[current] = {"texture": None, "color": [1.0, 1.0, 1.0],
                                      "alpha": 1.0}
            elif line.startswith("Kd ") and current:
                parts = line.split()[1:]
                materials[current]["color"] = [float(parts[0]), float(parts[1]), float(parts[2])]
            elif line.startswith("map_Kd ") and current:
                materials[current]["texture"] = line.split(" ", 1)[1].strip().replace("\\", "/")
            elif line.lower().startswith("map_bump ") and current:
                materials[current]["bump"] = line.split(" ", 1)[1].strip().replace("\\", "/")
            elif line.startswith("d ") and current:
                d = float(line.split()[1])
                if d < 0.999:
                    materials[current]["alpha"] = d
            elif line.startswith("illum 4") and current:
                # illum 4 is the "transparency: glass on" model. The .mtl's Tf
                # values are mush, so glass gets one flat alpha.
                if materials[current]["alpha"] >= 0.999:
                    materials[current]["alpha"] = 0.4
    return materials


def parse_obj(obj_path):
    """One trimesh pass over the OBJ into the flat arrays the demo needs.

    Returns a dict of numpy arrays: interleaved vertices (pos3+normal3+uv2),
    indices, the per-submesh draw table with AABBs and material slots, and the
    material slot table (a texture path relative to the OBJ, or "" plus a Kd
    colour). Runs once; the caller caches the result.
    """
    import trimesh  # the cached path never pays this import

    obj_dir = os.path.dirname(obj_path)
    mtl = load_materials(obj_path.replace(".obj", ".mtl"))
    print("Parsing the OBJ with trimesh. The first run takes minutes; "
          "the result is cached to a .npz next to the model.")
    scene = trimesh.load(obj_path, process=False)

    slots = {}          # key -> slot index
    tex_paths, kd_colors, kd_alphas, bump_paths = [], [], [], []

    def bump_for(mat_info):
        rel = (mat_info or {}).get("bump")
        if not rel:
            return ""  # normpath("") would be "." — a directory that "exists"
        rel = os.path.normpath(rel)
        return rel if os.path.isfile(os.path.join(obj_dir, rel)) else ""

    def slot_for(mat_info):
        if mat_info and mat_info["texture"]:
            rel = os.path.normpath(mat_info["texture"])
            if os.path.exists(os.path.join(obj_dir, rel)):
                key = ("tex", rel)
                if key not in slots:
                    slots[key] = len(tex_paths)
                    tex_paths.append(rel)
                    kd_colors.append([1.0, 1.0, 1.0])
                    kd_alphas.append(1.0)
                    bump_paths.append(bump_for(mat_info))
                return slots[key]
        color = mat_info["color"] if mat_info else [1.0, 1.0, 1.0]
        alpha = mat_info["alpha"] if mat_info else 1.0
        key = ("kd", tuple(round(c, 3) for c in color), round(alpha, 3))
        if key not in slots:
            slots[key] = len(tex_paths)
            tex_paths.append("")
            kd_colors.append(list(color))
            kd_alphas.append(alpha)
            bump_paths.append(bump_for(mat_info))
        return slots[key]

    verts, faces = [], []
    index_count, first_index, vertex_offset, tex_slot = [], [], [], []
    aabb_min, aabb_max = [], []
    v_off, i_off = 0, 0

    for name, geom in scene.geometry.items():
        v = np.asarray(geom.vertices, dtype=np.float32)
        f = np.asarray(geom.faces, dtype=np.uint32)
        if len(f) == 0 or len(v) == 0:
            continue

        n = getattr(geom, "vertex_normals", None)
        if n is None or len(n) != len(v):
            geom.fix_normals()
            n = geom.vertex_normals
        if n is None or len(n) != len(v):
            n = np.zeros_like(v)
            n[:, 1] = 1.0

        if hasattr(geom.visual, "uv") and geom.visual.uv is not None and len(geom.visual.uv) == len(v):
            uv = np.asarray(geom.visual.uv, dtype=np.float32).copy()
            # OBJ puts v=0 at the BOTTOM of the image; Vulkan samples row 0 at
            # the top. Without this flip every texture stands on its head.
            uv[:, 1] = 1.0 - uv[:, 1]
        else:
            uv = np.zeros((len(v), 2), dtype=np.float32)

        mat_info = None
        if hasattr(geom.visual, "material") and hasattr(geom.visual.material, "name"):
            mat_info = mtl.get(geom.visual.material.name)

        block = np.empty((len(v), 8), dtype=np.float32)
        block[:, 0:3] = v
        block[:, 3:6] = n
        block[:, 6:8] = uv
        verts.append(block)
        faces.append(f)

        index_count.append(len(f) * 3)
        first_index.append(i_off)
        vertex_offset.append(v_off)
        tex_slot.append(slot_for(mat_info))
        aabb_min.append(v.min(axis=0))
        aabb_max.append(v.max(axis=0))

        v_off += len(v)
        i_off += len(f) * 3

    return {
        "verts": np.concatenate(verts),
        "indices": np.concatenate([f.reshape(-1) for f in faces]).astype(np.uint32),
        "index_count": np.array(index_count, dtype=np.uint32),
        "first_index": np.array(first_index, dtype=np.uint32),
        "vertex_offset": np.array(vertex_offset, dtype=np.int32),
        "tex_slot": np.array(tex_slot, dtype=np.uint32),
        "aabb_min": np.array(aabb_min, dtype=np.float32),
        "aabb_max": np.array(aabb_max, dtype=np.float32),
        "tex_paths": np.array(tex_paths),
        "kd_colors": np.array(kd_colors, dtype=np.float32),
        "kd_alphas": np.array(kd_alphas, dtype=np.float32),
        "bump_paths": np.array(bump_paths),
    }


def load_scene_data(obj_path):
    """The npz cache around parse_obj: version + the OBJ's stat as the key."""
    cache_path = os.path.join(os.path.dirname(obj_path), "san-miguel.showcase.npz")
    stat = os.stat(obj_path)
    obj_stat = np.array([stat.st_mtime_ns, stat.st_size], dtype=np.int64)

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if (int(data["version"]) == CACHE_VERSION
                and np.array_equal(data["obj_stat"], obj_stat)):
            return {k: data[k] for k in data.files}
        print("The scene cache is stale (format or OBJ changed) — reparsing.")

    data = parse_obj(obj_path)
    np.savez(cache_path, version=np.int64(CACHE_VERSION), obj_stat=obj_stat, **data)
    print(f"Cached to {os.path.basename(cache_path)}.")
    return data


class DemoApp:
    def __init__(self):
        self.logger = bz.Logger()
        self.logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))
        self.window = bz.Window(W, H, "Bazalt Demo - Showcase", logger=self.logger)
        self.ctx = bz.Context(self.logger,
                              optional=[bz.Feature.BINDLESS, bz.Feature.MULTI_DRAW_INDIRECT],
                              hot_reload=True, gpu_timing=True)
        if not self.ctx.supports(bz.Feature.BINDLESS):
            raise SystemExit("this GPU reports no descriptorIndexing — "
                             "the material array cannot be built")
        if not self.ctx.supports(bz.Feature.MULTI_DRAW_INDIRECT):
            raise SystemExit("this GPU cannot read more than one indirect command — "
                             "the per-submesh culling has nothing to feed")

        self.renderer = self.ctx.create_renderer(self.window)
        self.window.set_cursor_mode(bz.CURSOR_DISABLED)
        self.camera = Camera()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.normpath(os.path.join(script_dir, "..", "assets"))
        obj_path = os.path.join(assets_dir, "San_Miguel", "san-miguel.obj")
        if not os.path.exists(obj_path):
            raise SystemExit("San Miguel is missing — run "
                             "examples/assets/download_san_miguel.py first")

        self.load_scene(obj_path)
        self.create_targets()
        self.create_pipelines(script_dir)
        self.create_descriptors()

        # One command buffer per frame slot. The recording changes every frame
        # (push constants carry the matrices), and a ring is what lets the
        # per-pass timers be read back: Timer.ms raises StateError once its
        # command buffer is re-recorded, so each slot's timers are read just
        # before that slot records again — its submit finished long ago.
        self.n_slots = self.ctx.frames_in_flight
        self.cmds = [self.ctx.create_command_buffer() for _ in range(self.n_slots)]
        self.slot_timers = [None] * self.n_slots
        self.pass_ms = {}
        self.timing_ok = True

        # Day-night state.
        self.day_time = 9.0
        self.time_scale = 1.0
        self.paused = False
        self.mode_index = 0
        self.last_visible = 0

    # ── scene ──────────────────────────────────────────────────────────────

    def load_scene(self, obj_path):
        ctx = self.ctx
        data = load_scene_data(obj_path)
        obj_dir = os.path.dirname(obj_path)

        self.submesh_count = len(data["index_count"])
        n = self.submesh_count

        self.vbuf = ctx.create_buffer(data["verts"].reshape(-1),
                                      bz.BufferType.VERTEX, bz.MemoryUsage.STATIC,
                                      name="scene vertices")
        self.ibuf = ctx.create_buffer(data["indices"],
                                      bz.BufferType.INDEX, bz.MemoryUsage.STATIC,
                                      name="scene indices")

        # Per-submesh AABBs for the culling compute: {centre.xyzw, extents.xyzw}.
        boxes = np.zeros((n, 8), dtype=np.float32)
        boxes[:, 0:3] = (data["aabb_min"] + data["aabb_max"]) * 0.5
        boxes[:, 4:7] = (data["aabb_max"] - data["aabb_min"]) * 0.5
        self.submesh_ssbo = ctx.create_buffer(boxes.reshape(-1),
                                              bz.BufferType.STORAGE, bz.MemoryUsage.STATIC,
                                              name="submesh AABBs")

        # N pre-filled VkDrawIndexedIndirectCommands. firstInstance carries the
        # material slot; the compute pass rewrites ONLY instanceCount. STATIC,
        # not DYNAMIC — a compute shader writes it (example 28's lesson).
        cmd_dtype = np.dtype([("index_count", "<u4"), ("instance_count", "<u4"),
                              ("first_index", "<u4"), ("vertex_offset", "<i4"),
                              ("first_instance", "<u4")])
        cmds = np.zeros(n, dtype=cmd_dtype)
        cmds["index_count"] = data["index_count"]
        cmds["first_index"] = data["first_index"]
        cmds["vertex_offset"] = data["vertex_offset"]
        cmds["first_instance"] = data["tex_slot"]

        # Glass leaves the opaque path: its commands keep indexCount 0 there
        # (the cull compute may set instanceCount, a zero-index draw is free)
        # and a small static list draws it blended after the opaques.
        transparent = data["kd_alphas"][data["tex_slot"]] < 0.999
        opaque_cmds = cmds.copy()
        opaque_cmds["index_count"][transparent] = 0
        self.cull_args = ctx.create_buffer(np.frombuffer(opaque_cmds.tobytes(), dtype=np.uint8),
                                           bz.BufferType.STORAGE, bz.MemoryUsage.STATIC,
                                           name="culled draw args")
        glass_cmds = cmds[transparent].copy()
        glass_cmds["instance_count"] = 1
        self.glass_count = len(glass_cmds)
        self.glass_args = ctx.create_buffer(
            np.frombuffer(glass_cmds.tobytes(), dtype=np.uint8) if self.glass_count
            else 20, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC, name="glass draw args")
        # The shadow pass draws everything: the light window's depth range
        # spans the scene, so culling would save nothing. Same commands, all
        # live — the shadow shader's alpha discard drops the glass.
        cmds["instance_count"] = 1
        self.shadow_args = ctx.create_buffer(np.frombuffer(cmds.tobytes(), dtype=np.uint8),
                                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC,
                                             name="shadow draw args")

        self.vis_counter = ctx.create_buffer(4, bz.BufferType.STORAGE,
                                             bz.MemoryUsage.STATIC, name="visible counter")

        # Materials: one bindless slot per unique texture, and a 1x1 image for
        # each texture-less Kd colour (cheaper than the vertex-colour channel
        # example 07 tiles over every vertex).
        self.textures = []
        for i, (path, kd, alpha) in enumerate(zip(data["tex_paths"], data["kd_colors"],
                                                  data["kd_alphas"])):
            if path:
                self.textures.append(ctx.load_image(os.path.join(obj_dir, str(path)),
                                                    name=os.path.basename(str(path))))
            else:
                # The material's transparency rides in this pixel's alpha; the
                # glass pass blends with it and the shadow pass discards on it.
                pixel = np.array([[[int(kd[0] * 255), int(kd[1] * 255),
                                    int(kd[2] * 255), int(alpha * 255)]]], dtype=np.uint8)
                self.textures.append(ctx.create_image(pixel, name=f"kd colour {i}"))
        self.normal_maps = self.load_normal_maps(obj_dir, data["bump_paths"])
        print(f"{n} submeshes ({self.glass_count} glass), {len(self.textures)} material "
              f"slots, {sum(1 for p in data['bump_paths'] if p)} bump maps.")

        # World bounds: the light's ortho fits these corners every frame, and
        # the fireflies roam a shrunken box near the ground.
        mn = data["aabb_min"].min(axis=0)
        mx = data["aabb_max"].max(axis=0)
        self.scene_center = glm.vec3(*((mn + mx) * 0.5))
        self.scene_radius = float(np.linalg.norm(mx - mn)) * 0.5
        self.aabb_corners = [glm.vec3(x, y, z)
                             for x in (mn[0], mx[0])
                             for y in (mn[1], mx[1])
                             for z in (mn[2], mx[2])]

        span = mx - mn
        lo = mn + span * 0.2
        hi = mx - span * 0.2
        lo[1] = mn[1] + 0.5
        hi[1] = min(mn[1] + 5.0, mx[1])
        self.firefly_lo, self.firefly_hi = lo, hi

        flies = np.zeros((FIREFLY_COUNT, 8), dtype=np.float32)
        rng = np.random.default_rng(7)
        flies[:, 0:3] = rng.uniform(lo, hi, (FIREFLY_COUNT, 3))
        flies[:, 3] = rng.uniform(0.0, math.tau, FIREFLY_COUNT)
        flies[:, 4:7] = rng.uniform(-0.3, 0.3, (FIREFLY_COUNT, 3))
        flies[:, 7] = rng.uniform(0.0, 100.0, FIREFLY_COUNT)
        self.firefly_buf = ctx.create_buffer(flies.reshape(-1),
                                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC,
                                             name="fireflies")

        self.frame_ubo = ctx.create_buffer(208, bz.BufferType.UNIFORM,
                                           bz.MemoryUsage.DYNAMIC, name="frame UBO")

    def load_normal_maps(self, obj_dir, bump_paths):
        """One normal map per material slot, flat where the .mtl has none.

        These go through PIL and create_image, NOT load_image: load_image
        decodes as sRGB, which would bend the vectors. Grayscale bump maps
        become normal maps through a height gradient.
        """
        from PIL import Image as PILImage

        flat = self.ctx.create_image(np.array([[[128, 128, 255, 255]]], dtype=np.uint8),
                                     name="flat normal")
        loaded = {}
        maps = []
        for path in bump_paths:
            p = str(path)
            if not p or not os.path.isfile(os.path.join(obj_dir, p)):
                maps.append(flat)
                continue
            if p not in loaded:
                img = np.asarray(PILImage.open(os.path.join(obj_dir, p)).convert("RGB"),
                                 dtype=np.uint8)
                spread = np.abs(img.astype(np.int16) - img[:, :, :1]).max()
                if spread < 8:  # grayscale height map — derive the normal
                    h = img[:, :, 0].astype(np.float32) / 255.0
                    gy, gx = np.gradient(h)
                    nrm = np.dstack([-gx * 4.0, -gy * 4.0, np.ones_like(h)])
                    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
                    img = ((nrm * 0.5 + 0.5) * 255.0).astype(np.uint8)
                rgba = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
                loaded[p] = self.ctx.create_image(rgba, mipmaps=True,
                                                  name=os.path.basename(p))
            maps.append(loaded[p])
        return maps

    # ── GPU objects ────────────────────────────────────────────────────────

    def create_targets(self):
        ctx = self.ctx
        self.samples = min(4, ctx.max_samples())
        # HDR scene target. samples>1 resolves both colour and depth
        # automatically, so .color[0] and .depth stay sampleable.
        self.scene_rt = ctx.create_render_target(W, H, color=bz.Format.RGBA16F,
                                                 depth=bz.Format.D32F,
                                                 samples=self.samples, name="scene HDR")
        self.shadow_rt = ctx.create_render_target(SHADOW_SIZE, SHADOW_SIZE,
                                                  color=None, depth=bz.Format.D32F,
                                                  name="shadow map")
        hw, hh = W // 2, H // 2
        # MRT: [0] bloom bright-pass, [1] god-ray mask.
        self.prepass_rt = ctx.create_render_target(hw, hh,
                                                   color=[bz.Format.RGBA16F, bz.Format.RGBA16F],
                                                   name="post prepass")
        self.blur_b_rt = ctx.create_render_target(hw, hh, color=bz.Format.RGBA16F,
                                                  name="bloom blur B")
        self.blur_a_rt = ctx.create_render_target(hw, hh, color=bz.Format.RGBA16F,
                                                  name="bloom blur A")
        self.godray_rt = ctx.create_render_target(hw, hh, color=bz.Format.RGBA16F,
                                                  name="god rays")

    def create_pipelines(self, script_dir):
        ctx = self.ctx

        def sh(name, stage):
            return ctx.compile_shader(os.path.join(script_dir, name), stage)

        fullscreen = sh("fullscreen.vert", bz.ShaderStage.VERTEX)
        vertex3 = [bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT2]
        tex_count = len(self.textures)

        self.cull_pipe = (ctx.compute_pipeline()
                          .shader(sh("cull.comp", bz.ShaderStage.COMPUTE))
                          .storage_buffer(0).storage_buffer(1).storage_buffer(2)
                          .push_constant(68)
                          .name("frustum cull")
                          .build())
        self.sim_pipe = (ctx.compute_pipeline()
                         .shader(sh("firefly.comp", bz.ShaderStage.COMPUTE))
                         .storage_buffer(0)
                         .push_constant(32)
                         .name("firefly sim")
                         .build())

        self.shadow_pipe = (ctx.graphics_pipeline()
                            .vertex_shader(sh("shadow.vert", bz.ShaderStage.VERTEX))
                            .fragment_shader(sh("shadow.frag", bz.ShaderStage.FRAGMENT))
                            .vertex_format(vertex3)
                            .depth_test(True)
                            .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                            .depth_bias(1.25, slope=1.75)
                            .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
                            # Both set=1 bindings, so the ONE bindless set is
                            # layout-compatible here and in the scene pipeline
                            # (the shadow shader only reads binding 0).
                            .texture(0, bz.ShaderStage.FRAGMENT, set=1, count=tex_count)
                            .texture(1, bz.ShaderStage.FRAGMENT, set=1, count=tex_count)
                            .name("shadow depth")
                            .build(self.shadow_rt))

        scene_vert = sh("scene.vert", bz.ShaderStage.VERTEX)
        scene_frag = sh("scene.frag", bz.ShaderStage.FRAGMENT)

        def scene_builder():
            return (ctx.graphics_pipeline()
                    .vertex_shader(scene_vert)
                    .fragment_shader(scene_frag)
                    .vertex_format(vertex3)
                    # NONE, not BACK: San Miguel's leaves are single-sided
                    # quads and must render from both sides (the shader flips
                    # the normal on gl_FrontFacing).
                    .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                    # Declaring one binding for two stages is two declarations
                    # — the builder ORs the stage flags of a repeated binding.
                    .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
                    .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                    .texture(1, bz.ShaderStage.FRAGMENT, set=0)
                    .storage_buffer(2, bz.ShaderStage.FRAGMENT, set=0)
                    .texture(0, bz.ShaderStage.FRAGMENT, set=1, count=tex_count)
                    .texture(1, bz.ShaderStage.FRAGMENT, set=1, count=tex_count))

        scene = scene_builder().depth_test(True).name("scene")
        if self.samples > 1:
            scene = scene.alpha_to_coverage(True)
        self.scene_pipe = scene.build(self.scene_rt)

        # Glass: the same shaders blended over the opaques. Depth test on so
        # walls hide it, write off so glass never occludes, no coverage
        # dithering — the alpha comes straight from the material's kd pixel.
        # Unsorted: San Miguel has a handful of vases, not a glass cathedral.
        self.glass_pipe = (scene_builder()
                          .depth_test(True, write=False)
                          .blend(True, mode=bz.BlendMode.ALPHA)
                          .name("glass")
                          .build(self.scene_rt))

        self.sky_pipe = (ctx.graphics_pipeline()
                         .vertex_shader(fullscreen)
                         .fragment_shader(sh("sky.frag", bz.ShaderStage.FRAGMENT))
                         .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                         .push_constant(64, bz.ShaderStage.FRAGMENT)
                         .name("sky")
                         .build(self.scene_rt))

        self.firefly_pipe = (ctx.graphics_pipeline()
                             .vertex_shader(sh("firefly.vert", bz.ShaderStage.VERTEX))
                             .fragment_shader(sh("firefly.frag", bz.ShaderStage.FRAGMENT))
                             .vertex_format([bz.VertexFormat.FLOAT4, bz.VertexFormat.FLOAT4])
                             .topology(bz.Topology.POINT_LIST)
                             .depth_test(True, write=False)
                             .blend(True, mode=bz.BlendMode.ADDITIVE)
                             .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
                             .name("firefly sprites")
                             .build(self.scene_rt))

        def post(frag, push_size, name, textures, target):
            builder = (ctx.graphics_pipeline()
                       .vertex_shader(fullscreen)
                       .fragment_shader(sh(frag, bz.ShaderStage.FRAGMENT))
                       .push_constant(push_size, bz.ShaderStage.FRAGMENT)
                       .name(name))
            for binding in range(textures):
                builder = builder.texture(binding, bz.ShaderStage.FRAGMENT, set=0)
            return builder.build(target)

        self.prepass_pipe = post("prepass.frag", 16, "post prepass", 2, self.prepass_rt)
        # blur_a and blur_b share format and size, so one pipeline serves both
        # directions (the same rule that lets example 16 reuse one pipeline
        # across cube faces).
        self.blur_pipe = post("blur.frag", 16, "bloom blur", 1, self.blur_b_rt)
        self.godray_pipe = post("godray.frag", 16, "god rays", 1, self.godray_rt)
        self.composite_pipe = post("composite.frag", 32, "composite", 3, self.renderer)

    def create_descriptors(self):
        ctx = self.ctx
        # Explicit sizes: the automatic pool defaults are far below two
        # bindless arrays of ~240 samplers each.
        pool = self.pool = ctx.create_descriptor_pool(
            max_sets=16,
            textures=2 * len(self.textures) + 16,
            uniform_buffers=8,
            storage_buffers=8)

        linear = ctx.create_sampler(filter=bz.Filter.LINEAR, address_mode=bz.AddressMode.CLAMP)
        nearest = ctx.create_sampler(filter=bz.Filter.NEAREST, address_mode=bz.AddressMode.CLAMP)
        # LINEAR + compare = hardware PCF; the OPAQUE_WHITE border means
        # "outside the map is lit", so the shader needs no uv guard.
        pcf = ctx.create_sampler(filter=bz.Filter.LINEAR,
                                 address_mode=bz.AddressMode.CLAMP_TO_BORDER,
                                 compare=bz.CompareOp.LESS,
                                 border_color=bz.BorderColor.OPAQUE_WHITE)

        self.cull_set = pool.allocate_set(self.cull_pipe, set=0)
        self.cull_set.set_buffer(0, self.submesh_ssbo)
        self.cull_set.set_buffer(1, self.cull_args)
        self.cull_set.set_buffer(2, self.vis_counter)

        self.sim_set = pool.allocate_set(self.sim_pipe, set=0)
        self.sim_set.set_buffer(0, self.firefly_buf)

        self.shadow_set = pool.allocate_frame_set(self.shadow_pipe, set=0)
        self.shadow_set.set_buffer(0, self.frame_ubo)

        self.scene_set = pool.allocate_frame_set(self.scene_pipe, set=0)
        self.scene_set.set_buffer(0, self.frame_ubo)
        self.scene_set.set_image(1, self.shadow_rt.depth, sampler=pcf)
        self.scene_set.set_buffer(2, self.firefly_buf)

        self.sky_set = pool.allocate_frame_set(self.sky_pipe, set=0)
        self.sky_set.set_buffer(0, self.frame_ubo)

        self.firefly_set = pool.allocate_frame_set(self.firefly_pipe, set=0)
        self.firefly_set.set_buffer(0, self.frame_ubo)

        # One bindless set serves the shadow AND the scene pipeline — their
        # set=1 layouts are declared identically.
        self.bindless_set = pool.allocate_set(self.scene_pipe, set=1)
        for i, tex in enumerate(self.textures):
            self.bindless_set.set_image(0, tex, index=i)
        for i, tex in enumerate(self.normal_maps):
            self.bindless_set.set_image(1, tex, index=i)

        self.prepass_set = pool.allocate_set(self.prepass_pipe, set=0)
        self.prepass_set.set_image(0, self.scene_rt.color[0], sampler=linear)
        self.prepass_set.set_image(1, self.scene_rt.depth, sampler=nearest)

        self.blur_h_set = pool.allocate_set(self.blur_pipe, set=0)
        self.blur_h_set.set_image(0, self.prepass_rt.color[0], sampler=linear)
        self.blur_v_set = pool.allocate_set(self.blur_pipe, set=0)
        self.blur_v_set.set_image(0, self.blur_b_rt.color[0], sampler=linear)

        self.godray_set = pool.allocate_set(self.godray_pipe, set=0)
        self.godray_set.set_image(0, self.prepass_rt.color[1], sampler=linear)

        self.composite_set = pool.allocate_set(self.composite_pipe, set=0)
        self.composite_set.set_image(0, self.scene_rt.color[0], sampler=linear)
        self.composite_set.set_image(1, self.blur_a_rt.color[0], sampler=linear)
        self.composite_set.set_image(2, self.godray_rt.color[0], sampler=linear)

    # ── the day-night model (CPU side) ─────────────────────────────────────

    def sun_direction(self):
        phi = (self.day_time - 6.0) / 12.0 * math.pi  # 0 at 06:00, pi at 18:00
        return glm.normalize(glm.vec3(math.cos(phi) * math.cos(SUN_AZIMUTH),
                                      math.sin(phi),
                                      math.cos(phi) * math.sin(SUN_AZIMUTH)))

    def light_matrix(self, light_dir):
        """An ortho that FOLLOWS the camera instead of covering the scene.

        A fixed SHADOW_RADIUS box around a point ahead of the camera keeps
        the texel density high wherever the player looks — one map does what
        a whole-scene fit needs cascades for. The depth range still spans the
        scene AABB, so casters outside the box (a wall between the light and
        the box) still occlude. Geometry past the box samples the map's
        OPAQUE_WHITE border and renders lit. The window is snapped to the
        texel grid, so walking does not make the shadow edges shimmer.
        """
        focus = self.camera.pos + self.camera.front * (SHADOW_RADIUS * 0.5)
        eye = focus + light_dir * self.scene_radius
        up = glm.vec3(0, 1, 0) if abs(light_dir.y) < 0.95 else glm.vec3(1, 0, 0)
        view = glm.lookAt(eye, focus, up)
        lo_z = 1e9
        hi_z = -1e9
        for corner in self.aabb_corners:
            z = (view * glm.vec4(corner, 1.0)).z
            lo_z = min(lo_z, z)
            hi_z = max(hi_z, z)
        r = SHADOW_RADIUS
        # The usual `proj[1][1] *= -1` flips the scale but not the offset —
        # swapping bottom and top flips y correctly for any window.
        proj = glm.orthoRH_ZO(-r, r, r, -r, -hi_z, -lo_z)
        lvp = proj * view
        # Texel snap: shift the window so the world origin lands on a texel
        # centre. Ortho w is 1, so a clip-space translate is exact.
        origin = lvp * glm.vec4(0.0, 0.0, 0.0, 1.0)
        half = SHADOW_SIZE / 2.0
        snap = glm.vec3(round(origin.x * half) / half - origin.x,
                        round(origin.y * half) / half - origin.y, 0.0)
        return glm.translate(glm.mat4(1.0), snap) * lvp

    def day_night(self, now):
        """Everything the UBO and the post pushes need for this time of day."""
        sun = self.sun_direction()
        moon = -sun
        night = 1.0 - smoothstep(-0.08, 0.05, sun.y)

        if sun.y > 0.0:
            light_dir, elevation = sun, sun.y
            warm = 1.0 - clamp(sun.y / 0.35, 0.0, 1.0)
            rgb = [1.0 - 0.0 * warm, 0.97 - 0.42 * warm, 0.92 - 0.67 * warm]
            intensity = 2.6 * smoothstep(0.0, 0.10, sun.y)
        else:
            light_dir, elevation = moon, moon.y
            rgb = [0.55, 0.65, 0.90]
            intensity = 0.30 * smoothstep(0.0, 0.10, moon.y)
        light_rgb = [c * intensity for c in rgb]
        shadow_strength = smoothstep(0.03, 0.15, elevation)

        day_amb = np.array([0.32, 0.38, 0.50]) * clamp(sun.y * 3.0 + 0.35, 0.25, 1.0) * 0.5
        ambient = day_amb * (1.0 - night) + np.array([0.020, 0.028, 0.055]) * night

        return {
            "sun": sun, "night": night, "light_dir": light_dir,
            "light_rgb": light_rgb, "shadow_strength": shadow_strength,
            "ambient": ambient, "time": now,
            "exposure": 1.1 + 0.9 * night,
            "warmth": (1.0 - night) * (1.0 - clamp(sun.y / 0.3, 0.0, 1.0)) ** 2 * 0.6,
        }

    def sun_screen(self, view_proj, sun):
        """The sun's uv position and a 0..1 visibility for god rays and flare."""
        clip = view_proj * glm.vec4(self.camera.pos + sun * 500.0, 1.0)
        if clip.w <= 0.0:
            return (0.5, 0.5), 0.0
        ndc_x, ndc_y = clip.x / clip.w, clip.y / clip.w
        vis = (clamp((1.0 - abs(ndc_x)) / 0.3, 0.0, 1.0)
               * clamp((1.0 - abs(ndc_y)) / 0.3, 0.0, 1.0)
               * smoothstep(-0.02, 0.06, sun.y))
        return (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5), vis

    # ── the frame ──────────────────────────────────────────────────────────

    def record(self, cmd, view_proj, dn, sun_uv, sun_vis, dt, now):
        cmd.begin()
        timers = {}

        with cmd.label("cull"):
            with cmd.timer() as t:
                cmd.fill_buffer(self.vis_counter, 0)
                (cmd.bind_pipeline(self.cull_pipe)
                    .bind_descriptor_set(self.cull_set, self.cull_pipe, set=0)
                    .push_constants(self.cull_pipe, 0,
                                    bytes(glm.transpose(view_proj))
                                    + struct.pack("<I", self.submesh_count))
                    .dispatch((self.submesh_count + 63) // 64))
            timers["cull"] = t

        with cmd.label("fireflies"):
            lo, hi = self.firefly_lo, self.firefly_hi
            (cmd.bind_pipeline(self.sim_pipe)
                .bind_descriptor_set(self.sim_set, self.sim_pipe, set=0)
                .push_constants(self.sim_pipe, 0,
                                struct.pack("<8f", lo[0], lo[1], lo[2], dt,
                                            hi[0], hi[1], hi[2], now))
                .dispatch((FIREFLY_COUNT + 63) // 64))

        with cmd.label("shadow"):
            with cmd.timer() as t:
                with cmd.rendering(self.shadow_rt) as c:
                    (c.bind_pipeline(self.shadow_pipe)
                      .bind_descriptor_set(self.shadow_set, self.shadow_pipe, set=0)
                      .bind_descriptor_set(self.bindless_set, self.shadow_pipe, set=1)
                      .bind_vertex_buffer(self.vbuf)
                      .bind_index_buffer(self.ibuf)
                      .draw_indexed_indirect(self.shadow_args, count=self.submesh_count))
            timers["shadow"] = t

        with cmd.label("scene"):
            with cmd.timer() as t:
                # ONE rendering scope: an MSAA target refuses clear_color=None,
                # so sky, geometry and fireflies are pipeline switches, not
                # passes.
                with cmd.rendering(self.scene_rt, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
                    (c.bind_pipeline(self.sky_pipe)
                      .bind_descriptor_set(self.sky_set, self.sky_pipe, set=0)
                      .push_constants(self.sky_pipe, 0, self.sky_push())
                      .draw(3))
                    (c.bind_pipeline(self.scene_pipe)
                      .bind_descriptor_set(self.scene_set, self.scene_pipe, set=0)
                      .bind_descriptor_set(self.bindless_set, self.scene_pipe, set=1)
                      .bind_vertex_buffer(self.vbuf)
                      .bind_index_buffer(self.ibuf)
                      .draw_indexed_indirect(self.cull_args, count=self.submesh_count))
                    if self.glass_count:
                        (c.bind_pipeline(self.glass_pipe)
                          .bind_descriptor_set(self.scene_set, self.glass_pipe, set=0)
                          .bind_descriptor_set(self.bindless_set, self.glass_pipe, set=1)
                          .draw_indexed_indirect(self.glass_args, count=self.glass_count))
                    (c.bind_pipeline(self.firefly_pipe)
                      .bind_descriptor_set(self.firefly_set, self.firefly_pipe, set=0)
                      .bind_vertex_buffer(self.firefly_buf)
                      .draw(FIREFLY_COUNT))
            timers["scene"] = t

        hw, hh = W // 2, H // 2
        with cmd.label("post"):
            with cmd.timer() as t:
                with cmd.rendering(self.prepass_rt) as c:
                    (c.bind_pipeline(self.prepass_pipe)
                      .bind_descriptor_set(self.prepass_set, self.prepass_pipe, set=0)
                      .push_constants(self.prepass_pipe, 0, struct.pack("<4f", 1.2, 0, 0, 0))
                      .draw(3))
                with cmd.rendering(self.blur_b_rt) as c:
                    (c.bind_pipeline(self.blur_pipe)
                      .bind_descriptor_set(self.blur_h_set, self.blur_pipe, set=0)
                      .push_constants(self.blur_pipe, 0,
                                      struct.pack("<4f", 2.0 / hw, 0.0, 0, 0))
                      .draw(3))
                with cmd.rendering(self.blur_a_rt) as c:
                    (c.bind_pipeline(self.blur_pipe)
                      .bind_descriptor_set(self.blur_v_set, self.blur_pipe, set=0)
                      .push_constants(self.blur_pipe, 0,
                                      struct.pack("<4f", 0.0, 2.0 / hh, 0, 0))
                      .draw(3))
                with cmd.rendering(self.godray_rt) as c:
                    (c.bind_pipeline(self.godray_pipe)
                      .bind_descriptor_set(self.godray_set, self.godray_pipe, set=0)
                      .push_constants(self.godray_pipe, 0,
                                      struct.pack("<4f", sun_uv[0], sun_uv[1],
                                                  0.9 * sun_vis, 0.94))
                      .draw(3))
            timers["post"] = t

        with cmd.label("composite"):
            with cmd.rendering(self.renderer) as c:
                (c.bind_pipeline(self.composite_pipe)
                  .bind_descriptor_set(self.composite_set, self.composite_pipe, set=0)
                  .push_constants(self.composite_pipe, 0,
                                  struct.pack("<8f", sun_uv[0], sun_uv[1], sun_vis,
                                              dn["night"], dn["exposure"], dn["warmth"],
                                              0, 0))
                  .draw(3))

        return timers

    def sky_push(self):
        cam = self.camera
        right = glm.normalize(glm.cross(cam.front, glm.vec3(0.0, 1.0, 0.0)))
        up = glm.cross(right, cam.front)
        return struct.pack(
            "16f",
            right.x, right.y, right.z, 0.0,
            up.x, up.y, up.z, 0.0,
            cam.front.x, cam.front.y, cam.front.z, 0.0,
            math.tan(glm.radians(60.0) / 2.0), W / H, 0.0, 0.0)

    def update_ubo(self, view_proj, light_vp, dn, now):
        amb = dn["ambient"]
        ld = dn["light_dir"]
        sun = dn["sun"]
        tail = struct.pack(
            "<20f",
            ld.x, ld.y, ld.z, dn["shadow_strength"],
            dn["light_rgb"][0], dn["light_rgb"][1], dn["light_rgb"][2], dn["night"],
            float(amb[0]), float(amb[1]), float(amb[2]), now,
            self.camera.pos.x, self.camera.pos.y, self.camera.pos.z, float(FIREFLY_COUNT),
            sun.x, sun.y, sun.z, 40.0)
        self.frame_ubo.update(bytes(glm.transpose(view_proj))
                              + bytes(glm.transpose(light_vp)) + tail)

    def read_slot_timers(self, slot):
        """Fold the timers recorded frames_in_flight ago into smoothed times.

        Read BEFORE this slot's command buffer records again — after that,
        Timer.ms raises StateError by design.
        """
        timers = self.slot_timers[slot]
        if not timers or not self.timing_ok:
            return
        try:
            for name, t in timers.items():
                ms = t.ms
                if ms is not None:
                    self.pass_ms[name] = self.pass_ms.get(name, ms) * 0.9 + ms * 0.1
        except bz.UnsupportedError:
            self.timing_ok = False  # this GPU never answers; stop asking
        except bz.StateError:
            pass

    def handle_keys(self, dt):
        w = self.window
        if w.was_key_pressed(bz.KEY_P):
            self.paused = not self.paused
        if w.was_key_pressed(bz.KEY_UP):
            self.time_scale = min(self.time_scale * 2.0, 16.0)
        if w.was_key_pressed(bz.KEY_DOWN):
            self.time_scale = max(self.time_scale / 2.0, 0.25)
        if w.was_key_pressed(bz.KEY_F):
            self.mode_index = (self.mode_index + 1) % len(WINDOW_MODES)
            w.set_mode(WINDOW_MODES[self.mode_index])
        # Hold to scrub: three in-scene hours per real second.
        if w.is_key_pressed(bz.KEY_RIGHT):
            self.day_time += dt * 3.0
        if w.is_key_pressed(bz.KEY_LEFT):
            self.day_time -= dt * 3.0
        if not self.paused:
            self.day_time += dt * self.time_scale * 24.0 / DAY_SECONDS
        self.day_time %= 24.0

    def run(self):
        print(__doc__)
        last_time = time.time()
        start = last_time
        frame_index = 0
        frames = 0
        fps_timer = 0.0

        while self.window.is_open():
            bz.poll_events()
            if self.window.is_key_pressed(bz.KEY_ESCAPE):
                break

            self.ctx.begin_frame()
            if not self.renderer.acquire():
                continue

            now_wall = time.time()
            dt = min(now_wall - last_time, 0.1)
            last_time = now_wall
            now = now_wall - start

            self.handle_keys(dt)
            mouse = self.window.get_mouse_state()
            right = self.camera.update_mouse(mouse.dx, mouse.dy)
            self.camera.process_keyboard(self.window, dt, right)

            dn = self.day_night(now)
            view_proj = self.camera.view_proj()
            light_vp = self.light_matrix(dn["light_dir"])
            sun_uv, sun_vis = self.sun_screen(view_proj, dn["sun"])
            self.update_ubo(view_proj, light_vp, dn, now)

            slot = frame_index % self.n_slots
            self.read_slot_timers(slot)
            cmd = self.cmds[slot]
            self.slot_timers[slot] = self.record(cmd, view_proj, dn, sun_uv,
                                                 sun_vis, dt, now)
            self.renderer.present(cmd)
            frame_index += 1

            frames += 1
            fps_timer += dt
            if fps_timer >= 1.0:
                fps = frames / fps_timer
                self.last_visible = int(self.vis_counter.read(np.uint32)[0])
                hh, mm = int(self.day_time), int(self.day_time % 1.0 * 60)
                parts = [f"Bazalt Demo - Showcase | {hh:02d}:{mm:02d}",
                         f"{self.last_visible}/{self.submesh_count} draws",
                         f"{1000.0 / fps:.1f} ms/frame | {fps:.0f} FPS"]
                if self.pass_ms:
                    parts.append(" ".join(f"{k} {v:.1f}" for k, v in self.pass_ms.items()))
                try:
                    total = self.renderer.gpu_time_ms
                    if total is not None:
                        parts.append(f"GPU {total:.1f} ms")
                except (bz.UnsupportedError, bz.StateError):
                    pass
                self.window.set_title(" | ".join(parts))
                frames = 0
                fps_timer = 0.0


if __name__ == "__main__":
    DemoApp().run()
