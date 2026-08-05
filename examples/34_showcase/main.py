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
  * The scene renders into a floating-point (RGBA16F) target with MSAA and
    alpha-to-coverage. That buffer is internal HEADROOM, not an HDR display
    mode: light adds up past 1.0 so bloom has something to find, and ACES
    maps the result back into the ordinary 0-1 range the swapchain shows.
    An 8-bit target would clip the sun to white before the bloom pass ever
    saw it. MSAA refuses a preserving second pass, so sky, scene and
    fireflies are pipelines inside ONE rendering scope.
  * A post chain adds SSAO, bloom, god rays, ACES tone mapping, a
    time-of-day grade and a vignette. The god-ray mask falls out of the depth
    buffer: sky pixels keep the cleared depth. The AO needs no G-buffer
    either — view positions come back through the projection constants and
    the normal from their derivatives.
  * Fireflies fly at night. A compute pass integrates them, a point draw
    renders them, and the scene shader reads the same buffer as a list of
    small point lights.
  * Profiling is first class: cmd.label() around every pass, name= on every
    resource, cmd.timer() per pass and gpu_timing for the frame total. Open a
    capture in Nsight or RenderDoc and the frame reads like this docstring.

The work is split the way the frame is: SceneLoader owns the geometry and the
materials, DayNightCycle owns the sun, ShadowRenderer owns the depth pass and
the light matrix, Fireflies owns its buffer and both its pipelines, and
PostProcessor owns everything after the scene target. DemoApp only wires them
together and runs the loop.

The first start parses the 1.1 GB OBJ with trimesh (minutes) and caches the
result to a .npz next to it. Later starts skip trimesh and load in seconds.
Get the model with examples/assets/download_san_miguel.py. Expect roughly
1.5-2 GB of VRAM: ~300 mipmapped textures plus the render targets.

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

W, H = 1920, 1080            # internal resolution, fixed for the run
CAM_FOV = 90.0               # degrees; the sky and AO passes rebuild rays from these
CAM_NEAR, CAM_FAR = 0.1, 500.0
SHADOW_SIZE = 4096
SHADOW_RADIUS = 22.0         # the shadow map covers this box around the camera
FIREFLY_COUNT = 128
DAY_SECONDS = 240.0          # one full day-night cycle at speed 1
SUN_AZIMUTH = -0.55          # radians; fixed heading of the sun's arc
CACHE_VERSION = 4            # v4: flipped V, per-slot alpha (glass)

WINDOW_MODES = [bz.WindowMode.WINDOWED, bz.WindowMode.FRAMELESS,
                bz.WindowMode.FULLSCREEN, bz.WindowMode.FULLSCREEN_WINDOWED]

VERTEX_FORMAT = [bz.VertexFormat.FLOAT3,   # position
                 bz.VertexFormat.FLOAT3,   # normal
                 bz.VertexFormat.FLOAT2]   # uv


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def camera_basis(front):
    """right/up for a front vector, the convention examples 04-07 share."""
    right = glm.normalize(glm.cross(front, glm.vec3(0.0, 1.0, 0.0)))
    return right, glm.cross(right, front)


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

    def update(self, window, dt):
        mouse = window.get_mouse_state()
        self.yaw += mouse.dx * self.sensitivity
        self.pitch = clamp(self.pitch - mouse.dy * self.sensitivity,
                           -math.pi / 2 + 0.01, math.pi / 2 - 0.01)
        self.front = glm.normalize(glm.vec3(
            math.cos(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            math.sin(self.yaw) * math.cos(self.pitch)))
        right, self.up = camera_basis(self.front)

        v = self.speed * dt
        if window.is_key_pressed(bz.Key.W): self.pos += v * self.front
        if window.is_key_pressed(bz.Key.S): self.pos -= v * self.front
        if window.is_key_pressed(bz.Key.A): self.pos -= v * right
        if window.is_key_pressed(bz.Key.D): self.pos += v * right
        if window.is_key_pressed(bz.Key.SPACE): self.pos += v * self.up
        if window.is_key_pressed(bz.Key.LEFT_SHIFT): self.pos -= v * self.up

    def view_proj(self):
        view = glm.lookAt(self.pos, self.pos + self.front, self.up)
        proj = glm.perspectiveRH_ZO(glm.radians(CAM_FOV), W / H, CAM_NEAR, CAM_FAR)
        proj[1][1] *= -1
        return proj * view

    def sky_push(self):
        """The camera basis the sky shader turns into a world-space ray.

        The FOV here MUST be the one view_proj() uses. With the two out of
        step the sky projects at a different rate from the geometry, and the
        stars slide across the scene as the camera turns.
        """
        right, up = camera_basis(self.front)
        return struct.pack(
            "16f",
            right.x, right.y, right.z, 0.0,
            up.x, up.y, up.z, 0.0,
            self.front.x, self.front.y, self.front.z, 0.0,
            math.tan(glm.radians(CAM_FOV) / 2.0), W / H, 0.0, 0.0)


class DayNightCycle:
    """The sun, the moon and everything their position decides.

    Pure CPU state: the time of day, the keys that scrub it, and the light
    colour, shadow strength, ambient and grading that follow from it.
    """

    def __init__(self, hour=9.0):
        self.hour = hour
        self.time_scale = 1.0
        self.paused = False

    def update(self, window, dt):
        if window.was_key_pressed(bz.Key.P):
            self.paused = not self.paused
        if window.was_key_pressed(bz.Key.UP):
            self.time_scale = min(self.time_scale * 2.0, 16.0)
        if window.was_key_pressed(bz.Key.DOWN):
            self.time_scale = max(self.time_scale / 2.0, 0.25)
        # Hold to scrub: three in-scene hours per real second.
        if window.is_key_pressed(bz.Key.RIGHT):
            self.hour += dt * 3.0
        if window.is_key_pressed(bz.Key.LEFT):
            self.hour -= dt * 3.0
        if not self.paused:
            self.hour += dt * self.time_scale * 24.0 / DAY_SECONDS
        self.hour %= 24.0

    def sun_direction(self):
        phi = (self.hour - 6.0) / 12.0 * math.pi  # 0 at 06:00, pi at 18:00
        return glm.normalize(glm.vec3(math.cos(phi) * math.cos(SUN_AZIMUTH),
                                      math.sin(phi),
                                      math.cos(phi) * math.sin(SUN_AZIMUTH)))

    def state(self, now):
        """Everything the frame UBO and the composite need for this hour."""
        sun = self.sun_direction()
        moon = -sun
        night = 1.0 - smoothstep(-0.08, 0.05, sun.y)

        if sun.y > 0.0:
            light_dir, elevation = sun, sun.y
            warm = 1.0 - clamp(sun.y / 0.35, 0.0, 1.0)
            rgb = [1.0, 0.97 - 0.42 * warm, 0.92 - 0.67 * warm]
            intensity = 2.6 * smoothstep(0.0, 0.06, sun.y)
        else:
            light_dir, elevation = moon, moon.y
            rgb = [0.55, 0.65, 0.90]
            intensity = 0.30 * smoothstep(0.0, 0.06, moon.y)

        # The fade only has to cover the instant the light flips from the sun
        # to the moon. Reaching further up (it used to hold until 0.15, about
        # 8.5 degrees) switched the shadows off through the whole golden hour
        # — every wall lit from inside while the sun was behind it.
        shadow_strength = smoothstep(0.005, 0.03, elevation)

        day_amb = np.array([0.32, 0.38, 0.50]) * clamp(sun.y * 3.0 + 0.35, 0.25, 1.0) * 0.5
        ambient = day_amb * (1.0 - night) + np.array([0.020, 0.028, 0.055]) * night

        return {
            "sun": sun,
            "night": night,
            "light_dir": light_dir,
            "light_rgb": [c * intensity for c in rgb],
            "shadow_strength": shadow_strength,
            "ambient": ambient,
            "exposure": 1.1 + 0.9 * night,
            "warmth": (1.0 - night) * (1.0 - clamp(sun.y / 0.3, 0.0, 1.0)) ** 2 * 0.6,
        }

    def sun_screen(self, view_proj, camera, sun):
        """The sun's uv position and a 0..1 visibility, for the god rays."""
        clip = view_proj * glm.vec4(camera.pos + sun * 500.0, 1.0)
        if clip.w <= 0.0:
            return (0.5, 0.5), 0.0
        ndc_x, ndc_y = clip.x / clip.w, clip.y / clip.w
        vis = (clamp((1.0 - abs(ndc_x)) / 0.3, 0.0, 1.0)
               * clamp((1.0 - abs(ndc_y)) / 0.3, 0.0, 1.0)
               * smoothstep(-0.02, 0.06, sun.y))
        return (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5), vis


class SceneLoader:
    """San Miguel: the OBJ parse, the .npz cache and the GPU resources.

    The parse runs once and caches; everything after it is buffers, images
    and the two indirect command lists the frame replays.
    """

    def __init__(self, ctx, obj_path):
        self.ctx = ctx
        obj_dir = os.path.dirname(obj_path)
        data = self.load_cached(obj_path)

        self.submesh_count = len(data["index_count"])
        self.build_geometry(data)
        self.build_draw_commands(data)
        self.build_materials(obj_dir, data)
        self.compute_bounds(data)
        print(f"{self.submesh_count} submeshes ({self.glass_count} glass), "
              f"{len(self.textures)} material slots, "
              f"{sum(1 for p in data['bump_paths'] if p)} bump maps.")

    # ── the CPU side ───────────────────────────────────────────────────────

    @staticmethod
    def load_materials(mtl_path):
        """Kd / map_Kd / map_Bump / transparency per material (07's parser
        grown)."""
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
                    materials[current]["color"] = [float(parts[0]), float(parts[1]),
                                                   float(parts[2])]
                elif line.startswith("map_Kd ") and current:
                    materials[current]["texture"] = (
                        line.split(" ", 1)[1].strip().replace("\\", "/"))
                elif line.lower().startswith("map_bump ") and current:
                    materials[current]["bump"] = (
                        line.split(" ", 1)[1].strip().replace("\\", "/"))
                elif line.startswith("d ") and current:
                    d = float(line.split()[1])
                    if d < 0.999:
                        materials[current]["alpha"] = d
                elif line.startswith("illum 4") and current:
                    # illum 4 is the "transparency: glass on" model. The
                    # .mtl's Tf values are mush, so glass gets one flat alpha.
                    if materials[current]["alpha"] >= 0.999:
                        materials[current]["alpha"] = 0.4
        return materials

    @classmethod
    def parse_obj(cls, obj_path):
        """One trimesh pass over the OBJ into the flat arrays the demo needs.

        Returns numpy arrays: interleaved vertices (pos3+normal3+uv2),
        indices, the per-submesh draw table with AABBs and material slots,
        and the material slot table (a texture path relative to the OBJ, or
        "" plus a Kd colour and alpha).
        """
        import trimesh  # the cached path never pays this import

        obj_dir = os.path.dirname(obj_path)
        mtl = cls.load_materials(obj_path.replace(".obj", ".mtl"))
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

            if (hasattr(geom.visual, "uv") and geom.visual.uv is not None
                    and len(geom.visual.uv) == len(v)):
                uv = np.asarray(geom.visual.uv, dtype=np.float32).copy()
                # OBJ puts v=0 at the BOTTOM of the image; Vulkan samples row
                # 0 at the top. Without this flip every texture stands on its
                # head.
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

    @classmethod
    def load_cached(cls, obj_path):
        """The npz cache around parse_obj: version + the OBJ's stat as key."""
        cache_path = os.path.join(os.path.dirname(obj_path), "san-miguel.showcase.npz")
        stat = os.stat(obj_path)
        obj_stat = np.array([stat.st_mtime_ns, stat.st_size], dtype=np.int64)

        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            if (int(data["version"]) == CACHE_VERSION
                    and np.array_equal(data["obj_stat"], obj_stat)):
                return {k: data[k] for k in data.files}
            print("The scene cache is stale (format or OBJ changed) — reparsing.")

        data = cls.parse_obj(obj_path)
        np.savez(cache_path, version=np.int64(CACHE_VERSION), obj_stat=obj_stat, **data)
        print(f"Cached to {os.path.basename(cache_path)}.")
        return data

    # ── the GPU side ───────────────────────────────────────────────────────

    def build_geometry(self, data):
        self.vertices = self.ctx.create_buffer(
            data["verts"].reshape(-1),
            type=bz.BufferType.VERTEX, usage=bz.MemoryUsage.STATIC,
            name="scene vertices")
        self.indices = self.ctx.create_buffer(
            data["indices"],
            type=bz.BufferType.INDEX, usage=bz.MemoryUsage.STATIC,
            name="scene indices")

        # Per-submesh AABBs for the culling compute: {centre.xyzw, extents.xyzw}.
        boxes = np.zeros((self.submesh_count, 8), dtype=np.float32)
        boxes[:, 0:3] = (data["aabb_min"] + data["aabb_max"]) * 0.5
        boxes[:, 4:7] = (data["aabb_max"] - data["aabb_min"]) * 0.5
        self.submesh_boxes = self.ctx.create_buffer(
            boxes.reshape(-1),
            type=bz.BufferType.STORAGE, usage=bz.MemoryUsage.STATIC,
            name="submesh AABBs")

    def build_draw_commands(self, data):
        """Three indirect lists out of one table of VkDrawIndexedIndirectCommand.

        firstInstance carries the material slot; the cull compute rewrites
        ONLY instanceCount. STATIC, not DYNAMIC — a compute shader writes it
        (example 28's lesson).
        """
        cmd_dtype = np.dtype([("index_count", "<u4"), ("instance_count", "<u4"),
                              ("first_index", "<u4"), ("vertex_offset", "<i4"),
                              ("first_instance", "<u4")])
        cmds = np.zeros(self.submesh_count, dtype=cmd_dtype)
        cmds["index_count"] = data["index_count"]
        cmds["first_index"] = data["first_index"]
        cmds["vertex_offset"] = data["vertex_offset"]
        cmds["first_instance"] = data["tex_slot"]

        def as_bytes(array):
            return np.frombuffer(array.tobytes(), dtype=np.uint8)

        # Glass leaves the opaque path: its commands keep indexCount 0 there
        # (the cull compute may still set instanceCount, a zero-index draw is
        # free) and a small static list draws it blended after the opaques.
        transparent = data["kd_alphas"][data["tex_slot"]] < 0.999
        opaque = cmds.copy()
        opaque["index_count"][transparent] = 0
        self.cull_args = self.ctx.create_buffer(
            as_bytes(opaque),
            type=bz.BufferType.STORAGE, usage=bz.MemoryUsage.STATIC,
            name="culled draw args")

        glass = cmds[transparent].copy()
        glass["instance_count"] = 1
        self.glass_count = len(glass)
        self.glass_args = self.ctx.create_buffer(
            as_bytes(glass) if self.glass_count else 20,
            type=bz.BufferType.STORAGE, usage=bz.MemoryUsage.STATIC,
            name="glass draw args")

        # The shadow pass draws everything: the light window's depth range
        # spans the scene, so culling would save nothing. Same commands, all
        # live — the shadow shader's alpha discard drops the glass.
        cmds["instance_count"] = 1
        self.shadow_args = self.ctx.create_buffer(
            as_bytes(cmds),
            type=bz.BufferType.STORAGE, usage=bz.MemoryUsage.STATIC,
            name="shadow draw args")

        self.visible_counter = self.ctx.create_buffer(
            4, type=bz.BufferType.STORAGE, usage=bz.MemoryUsage.STATIC,
            name="visible counter")

    def build_materials(self, obj_dir, data):
        """One bindless slot per unique material: a diffuse and a normal map.

        A texture-less Kd colour becomes a 1x1 image, which is cheaper than
        the vertex-colour channel example 07 tiles over every vertex, and
        carries the material's transparency in its alpha.
        """
        self.textures = []
        for i, (path, kd, alpha) in enumerate(zip(data["tex_paths"], data["kd_colors"],
                                                  data["kd_alphas"])):
            if path:
                self.textures.append(self.ctx.load_image(
                    os.path.join(obj_dir, str(path)),
                    name=os.path.basename(str(path))))
            else:
                pixel = np.array([[[int(kd[0] * 255), int(kd[1] * 255),
                                    int(kd[2] * 255), int(alpha * 255)]]], dtype=np.uint8)
                self.textures.append(self.ctx.create_image(pixel, name=f"kd colour {i}"))
        self.normal_maps = self.load_normal_maps(obj_dir, data["bump_paths"])

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

    def compute_bounds(self, data):
        """World bounds for the light's depth range and the firefly roam box."""
        mn = data["aabb_min"].min(axis=0)
        mx = data["aabb_max"].max(axis=0)
        self.center = glm.vec3(*((mn + mx) * 0.5))
        self.radius = float(np.linalg.norm(mx - mn)) * 0.5
        self.corners = [glm.vec3(x, y, z)
                        for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])]

        span = mx - mn
        lo = mn + span * 0.2
        hi = mx - span * 0.2
        lo[1] = mn[1] + 0.5
        hi[1] = min(mn[1] + 5.0, mx[1])
        self.firefly_box = (lo, hi)

    def bind_geometry(self, c):
        return c.bind_vertex_buffer(self.vertices).bind_index_buffer(self.indices)


class ShadowRenderer:
    """The depth-only pass and the light matrix that aims it."""

    def __init__(self, ctx, pool, scene, frame_ubo, shader):
        self.ctx = ctx
        self.scene = scene
        self.target = ctx.create_render_target(
            SHADOW_SIZE, SHADOW_SIZE, color=None, depth=bz.Format.D32F,
            name="shadow map")

        self.pipeline = (ctx.graphics_pipeline()
                         .vertex_shader(shader("shadow.vert", bz.ShaderStage.VERTEX))
                         .fragment_shader(shader("shadow.frag", bz.ShaderStage.FRAGMENT))
                         .vertex_format(VERTEX_FORMAT)
                         .depth_test(True)
                         .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                         .depth_bias(1.25, slope=1.75)
                         .uniform_buffer(binding=0, stage=bz.ShaderStage.VERTEX)
                         # Both set=1 bindings, so ONE bindless set is
                         # layout-compatible here and in the scene pipeline
                         # (this shader only reads binding 0).
                         .texture(binding=0, stage=bz.ShaderStage.FRAGMENT, set=1,
                                  count=len(scene.textures))
                         .texture(binding=1, stage=bz.ShaderStage.FRAGMENT, set=1,
                                  count=len(scene.textures))
                         .name("shadow depth")
                         .build(self.target))

        self.frame_set = pool.allocate_frame_set(self.pipeline, set=0)
        self.frame_set.set_buffer(0, frame_ubo)

    @property
    def depth(self):
        return self.target.depth

    def light_matrix(self, camera, light_dir):
        """An ortho that FOLLOWS the camera instead of covering the scene.

        A fixed SHADOW_RADIUS box around a point ahead of the camera keeps
        the texel density high wherever the player looks — one map does what
        a whole-scene fit needs cascades for. The depth range still spans the
        scene AABB, so casters outside the box (a wall between the light and
        the box) still occlude. Geometry past the box samples the map's
        OPAQUE_WHITE border and renders lit. The window is snapped to the
        texel grid, so walking does not make the shadow edges shimmer.
        """
        focus = camera.pos + camera.front * (SHADOW_RADIUS * 0.5)
        eye = focus + light_dir * self.scene.radius
        up = glm.vec3(0, 1, 0) if abs(light_dir.y) < 0.95 else glm.vec3(1, 0, 0)
        view = glm.lookAt(eye, focus, up)

        lo_z, hi_z = 1e9, -1e9
        for corner in self.scene.corners:
            z = (view * glm.vec4(corner, 1.0)).z
            lo_z = min(lo_z, z)
            hi_z = max(hi_z, z)

        r = SHADOW_RADIUS
        # The usual `proj[1][1] *= -1` flips the scale but not the offset —
        # swapping bottom and top flips y correctly for any window.
        proj = glm.orthoRH_ZO(-r, r, r, -r, -hi_z, -lo_z)
        light_vp = proj * view

        # Texel snap: shift the window so the world origin lands on a texel
        # centre. Ortho w is 1, so a clip-space translate is exact.
        origin = light_vp * glm.vec4(0.0, 0.0, 0.0, 1.0)
        half = SHADOW_SIZE / 2.0
        snap = glm.vec3(round(origin.x * half) / half - origin.x,
                        round(origin.y * half) / half - origin.y, 0.0)
        return glm.translate(glm.mat4(1.0), snap) * light_vp

    def record(self, cmd, bindless_set):
        with cmd.rendering(self.target) as c:
            (c.bind_pipeline(self.pipeline)
              .bind_descriptor_set(self.frame_set, self.pipeline, set=0)
              .bind_descriptor_set(bindless_set, self.pipeline, set=1))
            (self.scene.bind_geometry(c)
              .draw_indexed_indirect(self.scene.shadow_args,
                                     count=self.scene.submesh_count))


class Fireflies:
    """One buffer, three readers.

    A compute pass integrates {pos, vel}; a point draw renders the sprites
    straight out of the same buffer (STORAGE buffers carry VERTEX usage);
    and the scene shader reads it as a list of point lights.
    """

    def __init__(self, ctx, pool, scene, frame_ubo, scene_target, shader):
        self.ctx = ctx
        self.box = scene.firefly_box
        lo, hi = self.box

        rng = np.random.default_rng(7)
        flies = np.zeros((FIREFLY_COUNT, 8), dtype=np.float32)
        flies[:, 0:3] = rng.uniform(lo, hi, (FIREFLY_COUNT, 3))
        flies[:, 3] = rng.uniform(0.0, math.tau, FIREFLY_COUNT)   # flicker phase
        flies[:, 4:7] = rng.uniform(-0.3, 0.3, (FIREFLY_COUNT, 3))
        flies[:, 7] = rng.uniform(0.0, 100.0, FIREFLY_COUNT)      # seed
        self.buffer = ctx.create_buffer(
            flies.reshape(-1),
            type=bz.BufferType.STORAGE, usage=bz.MemoryUsage.STATIC,
            name="fireflies")

        self.sim_pipeline = (ctx.compute_pipeline()
                             .shader(shader("firefly.comp", bz.ShaderStage.COMPUTE))
                             .storage_buffer(binding=0)
                             .push_constant(32)
                             .name("firefly sim")
                             .build())
        self.draw_pipeline = (ctx.graphics_pipeline()
                              .vertex_shader(shader("firefly.vert", bz.ShaderStage.VERTEX))
                              .fragment_shader(shader("firefly.frag", bz.ShaderStage.FRAGMENT))
                              .vertex_format([bz.VertexFormat.FLOAT4,
                                              bz.VertexFormat.FLOAT4])
                              .topology(bz.Topology.POINT_LIST)
                              .depth_test(True, write=False)
                              .blend(True, mode=bz.BlendMode.ADDITIVE)
                              .uniform_buffer(binding=0, stage=bz.ShaderStage.VERTEX)
                              .name("firefly sprites")
                              .build(scene_target))

        self.sim_set = pool.allocate_set(self.sim_pipeline)
        self.sim_set.set_buffer(0, self.buffer)
        self.draw_set = pool.allocate_frame_set(self.draw_pipeline)
        self.draw_set.set_buffer(0, frame_ubo)

    def record_simulation(self, cmd, dt, now):
        lo, hi = self.box
        (cmd.bind_pipeline(self.sim_pipeline)
            .bind_descriptor_set(self.sim_set, self.sim_pipeline)
            .push_constants(self.sim_pipeline, offset=0,
                            data=struct.pack("<8f", lo[0], lo[1], lo[2], dt,
                                             hi[0], hi[1], hi[2], now))
            .dispatch((FIREFLY_COUNT + 63) // 64))

    def record_draw(self, c):
        (c.bind_pipeline(self.draw_pipeline)
          .bind_descriptor_set(self.draw_set, self.draw_pipeline)
          .bind_vertex_buffer(self.buffer)
          .draw(FIREFLY_COUNT))


class PostProcessor:
    """Everything after the scene target: SSAO, bloom, god rays, composite.

    All the intermediate targets are half resolution and share one format,
    which is what lets a single gaussian pipeline blur both the AO and the
    bloom.
    """

    BLOOM_THRESHOLD = 1.2
    BLOOM_CEILING = 6.0      # see prepass.frag: this is what keeps the sun round
    BLOOM_STEPS = (2.0, 6.0)  # texels per tap, one entry per gaussian iteration
    AO_RADIUS = 1.4          # metres
    AO_STRENGTH = 3.0

    def __init__(self, ctx, pool, scene_target, output_target, shader):
        self.ctx = ctx
        self.scene_target = scene_target
        self.output = output_target
        self.hw, self.hh = W // 2, H // 2

        def half(name, color=bz.Format.RGBA16F):
            return ctx.create_render_target(self.hw, self.hh, color=color, name=name)

        # MRT: [0] bloom bright-pass, [1] god-ray mask.
        self.prepass_rt = half("post prepass",
                               color=[bz.Format.RGBA16F, bz.Format.RGBA16F])
        self.blur_b_rt = half("bloom blur B")
        self.blur_a_rt = half("bloom blur A")
        self.godray_rt = half("god rays")
        self.ao_rt = half("SSAO")
        self.ao_tmp_rt = half("SSAO blur tmp")

        def post_pipeline(frag, push_size, textures, target, name):
            builder = (ctx.graphics_pipeline()
                       .vertex_shader(shader("fullscreen.vert", bz.ShaderStage.VERTEX))
                       .fragment_shader(shader(frag, bz.ShaderStage.FRAGMENT))
                       .push_constant(push_size, bz.ShaderStage.FRAGMENT)
                       .name(name))
            for binding in range(textures):
                builder = builder.texture(binding=binding,
                                          stage=bz.ShaderStage.FRAGMENT)
            return builder.build(target)

        self.prepass_pipe = post_pipeline("prepass.frag", 16, 2, self.prepass_rt,
                                          "post prepass")
        # blur_a and blur_b share format and size, so one pipeline serves
        # every direction and both effects (the same rule that lets example
        # 16 reuse one pipeline across cube faces).
        self.blur_pipe = post_pipeline("blur.frag", 16, 1, self.blur_b_rt, "gaussian blur")
        self.godray_pipe = post_pipeline("godray.frag", 16, 1, self.godray_rt, "god rays")
        self.ao_pipe = post_pipeline("ao.frag", 32, 1, self.ao_rt, "SSAO")
        self.composite_pipe = post_pipeline("composite.frag", 32, 4, output_target,
                                            "composite")

        linear = ctx.create_sampler(filter=bz.Filter.LINEAR,
                                    address_mode=bz.AddressMode.CLAMP)
        nearest = ctx.create_sampler(filter=bz.Filter.NEAREST,
                                     address_mode=bz.AddressMode.CLAMP)

        def source_set(pipeline, *images, sampler=linear):
            dset = pool.allocate_set(pipeline)
            for binding, image in enumerate(images):
                dset.set_image(binding, image, sampler=sampler)
            return dset

        self.prepass_set = pool.allocate_set(self.prepass_pipe)
        self.prepass_set.set_image(0, scene_target.color[0], sampler=linear)
        self.prepass_set.set_image(1, scene_target.depth, sampler=nearest)

        self.ao_set = source_set(self.ao_pipe, scene_target.depth, sampler=nearest)
        self.ao_blur_h_set = source_set(self.blur_pipe, self.ao_rt.color[0])
        self.ao_blur_v_set = source_set(self.blur_pipe, self.ao_tmp_rt.color[0])

        self.bloom_h_sets = [source_set(self.blur_pipe, self.prepass_rt.color[0]),
                             source_set(self.blur_pipe, self.blur_a_rt.color[0])]
        self.bloom_v_set = source_set(self.blur_pipe, self.blur_b_rt.color[0])

        self.godray_set = source_set(self.godray_pipe, self.prepass_rt.color[1])
        self.composite_set = source_set(self.composite_pipe,
                                        scene_target.color[0],
                                        self.blur_a_rt.color[0],
                                        self.godray_rt.color[0],
                                        self.ao_rt.color[0])

    def blur(self, cmd, source_set, target, dx, dy):
        with cmd.rendering(target) as c:
            (c.bind_pipeline(self.blur_pipe)
              .bind_descriptor_set(source_set, self.blur_pipe)
              .push_constants(self.blur_pipe, offset=0,
                              data=struct.pack("<4f", dx, dy, 0.0, 0.0))
              .draw(3))

    def fullscreen(self, cmd, pipeline, dset, target, push):
        with cmd.rendering(target) as c:
            (c.bind_pipeline(pipeline)
              .bind_descriptor_set(dset, pipeline)
              .push_constants(pipeline, offset=0, data=push)
              .draw(3))

    def record(self, cmd, day, sun_uv, sun_vis):
        hw, hh = self.hw, self.hh

        self.fullscreen(cmd, self.ao_pipe, self.ao_set, self.ao_rt,
                        struct.pack("<8f", math.tan(math.radians(CAM_FOV) / 2.0),
                                    W / H, CAM_NEAR, CAM_FAR,
                                    self.AO_RADIUS, self.AO_STRENGTH, 0.0, 0.0))
        self.blur(cmd, self.ao_blur_h_set, self.ao_tmp_rt, 1.0 / hw, 0.0)
        self.blur(cmd, self.ao_blur_v_set, self.ao_rt, 0.0, 1.0 / hh)

        self.fullscreen(cmd, self.prepass_pipe, self.prepass_set, self.prepass_rt,
                        struct.pack("<4f", self.BLOOM_THRESHOLD, self.BLOOM_CEILING,
                                    0.0, 0.0))
        # Two gaussian iterations, the second three times as wide: one 9-tap
        # pass is too narrow for a glow, and widening its step alone bands.
        for i, step in enumerate(self.BLOOM_STEPS):
            self.blur(cmd, self.bloom_h_sets[i], self.blur_b_rt, step / hw, 0.0)
            self.blur(cmd, self.bloom_v_set, self.blur_a_rt, 0.0, step / hh)

        self.fullscreen(cmd, self.godray_pipe, self.godray_set, self.godray_rt,
                        struct.pack("<4f", sun_uv[0], sun_uv[1], 0.9 * sun_vis, 0.94))

    def record_composite(self, cmd, day):
        self.fullscreen(cmd, self.composite_pipe, self.composite_set, self.output,
                        struct.pack("<8f", day["night"], day["exposure"],
                                    day["warmth"], 0.0, 0.0, 0.0, 0.0, 0.0))


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
        self.window.set_cursor_mode(bz.CursorMode.DISABLED)
        self.camera = Camera()
        self.day = DayNightCycle()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        obj_path = os.path.normpath(os.path.join(
            script_dir, "..", "assets", "San_Miguel", "san-miguel.obj"))
        if not os.path.exists(obj_path):
            raise SystemExit("San Miguel is missing — run "
                             "examples/assets/download_san_miguel.py first")

        def shader(name, stage):
            return self.ctx.compile_shader(os.path.join(script_dir, name), stage)

        self.scene = SceneLoader(self.ctx, obj_path)
        self.create_targets()
        self.frame_ubo = self.ctx.create_buffer(
            208, type=bz.BufferType.UNIFORM, usage=bz.MemoryUsage.DYNAMIC,
            name="frame UBO")
        self.create_pool()
        self.create_scene_pipelines(shader)

        self.shadows = ShadowRenderer(self.ctx, self.pool, self.scene,
                                      self.frame_ubo, shader)
        self.fireflies = Fireflies(self.ctx, self.pool, self.scene, self.frame_ubo,
                                   self.scene_rt, shader)
        self.post = PostProcessor(self.ctx, self.pool, self.scene_rt,
                                  self.renderer, shader)
        self.create_scene_sets()

        # One command buffer per frame slot. The recording changes every frame
        # (push constants carry the matrices), and a ring is what lets the
        # per-pass timers be read back: Timer.ms raises StateError once its
        # command buffer is re-recorded, so each slot's timers are read just
        # before that slot records again — its submit finished long ago.
        self.slot_count = self.ctx.frames_in_flight
        self.cmds = [self.ctx.create_command_buffer() for _ in range(self.slot_count)]
        self.slot_timers = [None] * self.slot_count
        self.pass_ms = {}
        self.timing_ok = True
        self.mode_index = 0
        self.last_visible = 0

    # ── setup ──────────────────────────────────────────────────────────────

    def create_targets(self):
        self.samples = min(4, self.ctx.max_samples())
        # samples>1 resolves both colour and depth automatically, so
        # .color[0] and .depth stay sampleable by the post chain.
        self.scene_rt = self.ctx.create_render_target(
            W, H, color=bz.Format.RGBA16F, depth=bz.Format.D32F,
            samples=self.samples, name="scene HDR")

    def create_pool(self):
        # The one hand-budgeted pool in the examples, so the escape hatch is
        # shown somewhere: explicit sizes mean one fixed block and a
        # ResourceError when it runs out. Everything else calls
        # create_descriptor_pool() with no arguments and lets it size itself.
        # A frame set costs frames_in_flight sets and that many of each
        # descriptor it holds, which is the arithmetic the automatic pool
        # exists to remove.
        frames = self.ctx.frames_in_flight
        self.pool = self.ctx.create_descriptor_pool(
            max_sets=4 * frames + 20,
            textures=2 * len(self.scene.textures) + 4 * frames + 20,
            uniform_buffers=4 * frames + 4,
            storage_buffers=2 * frames + 8)

    def create_scene_pipelines(self, shader):
        ctx = self.ctx
        tex_count = len(self.scene.textures)

        self.cull_pipe = (ctx.compute_pipeline()
                          .shader(shader("cull.comp", bz.ShaderStage.COMPUTE))
                          .storage_buffer(binding=0)
                          .storage_buffer(binding=1)
                          .storage_buffer(binding=2)
                          .push_constant(68)
                          .name("frustum cull")
                          .build())

        scene_vert = shader("scene.vert", bz.ShaderStage.VERTEX)
        scene_frag = shader("scene.frag", bz.ShaderStage.FRAGMENT)

        def scene_builder():
            # set=0 is per-frame (camera, shadow map, lights); set=1 is the
            # bindless material array, shared with the shadow pipeline. Every
            # other pipeline here has one set and says nothing about it.
            return (ctx.graphics_pipeline()
                    .vertex_shader(scene_vert)
                    .fragment_shader(scene_frag)
                    .vertex_format(VERTEX_FORMAT)
                    # NONE, not BACK: San Miguel's leaves are single-sided
                    # quads and must render from both sides (the shader flips
                    # the normal on gl_FrontFacing).
                    .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                    # Declaring one binding for two stages is two
                    # declarations — the builder ORs the stage flags.
                    .uniform_buffer(binding=0, stage=bz.ShaderStage.VERTEX)
                    .uniform_buffer(binding=0, stage=bz.ShaderStage.FRAGMENT)
                    .texture(binding=1, stage=bz.ShaderStage.FRAGMENT)
                    .storage_buffer(binding=2, stage=bz.ShaderStage.FRAGMENT)
                    .texture(binding=0, stage=bz.ShaderStage.FRAGMENT, set=1,
                             count=tex_count)
                    .texture(binding=1, stage=bz.ShaderStage.FRAGMENT, set=1,
                             count=tex_count))

        scene = scene_builder().depth_test(True).name("scene")
        if self.samples > 1:
            scene = scene.alpha_to_coverage(True)
        self.scene_pipe = scene.build(self.scene_rt)

        # Glass: the same shaders blended over the opaques. Depth test on so
        # walls hide it, write off so glass never occludes, and the alpha
        # sharpening switched off — its alpha is a flat number from the
        # material pixel and sharpening would drive it to zero.
        # Unsorted: San Miguel has a handful of vases, not a glass cathedral.
        self.glass_pipe = (scene_builder()
                           .depth_test(True, write=False)
                           .blend(True, mode=bz.BlendMode.ALPHA)
                           .constant(0, False, bz.ShaderStage.FRAGMENT)
                           .name("glass")
                           .build(self.scene_rt))

        self.sky_pipe = (ctx.graphics_pipeline()
                         .vertex_shader(shader("fullscreen.vert", bz.ShaderStage.VERTEX))
                         .fragment_shader(shader("sky.frag", bz.ShaderStage.FRAGMENT))
                         .uniform_buffer(binding=0, stage=bz.ShaderStage.FRAGMENT)
                         .push_constant(64, bz.ShaderStage.FRAGMENT)
                         .name("sky")
                         .build(self.scene_rt))

    def create_scene_sets(self):
        pcf = self.ctx.create_sampler(filter=bz.Filter.LINEAR,
                                      address_mode=bz.AddressMode.CLAMP_TO_BORDER,
                                      compare=bz.CompareOp.LESS,
                                      border_color=bz.BorderColor.OPAQUE_WHITE)

        self.cull_set = self.pool.allocate_set(self.cull_pipe)
        self.cull_set.set_buffer(0, self.scene.submesh_boxes)
        self.cull_set.set_buffer(1, self.scene.cull_args)
        self.cull_set.set_buffer(2, self.scene.visible_counter)

        self.scene_set = self.pool.allocate_frame_set(self.scene_pipe, set=0)
        self.scene_set.set_buffer(0, self.frame_ubo)
        self.scene_set.set_image(1, self.shadows.depth, sampler=pcf)
        self.scene_set.set_buffer(2, self.fireflies.buffer)

        self.sky_set = self.pool.allocate_frame_set(self.sky_pipe)
        self.sky_set.set_buffer(0, self.frame_ubo)

        # One bindless set serves the scene, the glass AND the shadow
        # pipeline — their set=1 layouts are declared identically.
        self.bindless_set = self.pool.allocate_set(self.scene_pipe, set=1)
        for i, image in enumerate(self.scene.textures):
            self.bindless_set.set_image(0, image, index=i)
        for i, image in enumerate(self.scene.normal_maps):
            self.bindless_set.set_image(1, image, index=i)

    # ── the frame ──────────────────────────────────────────────────────────

    def update_frame_ubo(self, view_proj, light_vp, day, now):
        amb = day["ambient"]
        ld = day["light_dir"]
        sun = day["sun"]
        tail = struct.pack(
            "<20f",
            ld.x, ld.y, ld.z, day["shadow_strength"],
            day["light_rgb"][0], day["light_rgb"][1], day["light_rgb"][2], day["night"],
            float(amb[0]), float(amb[1]), float(amb[2]), now,
            self.camera.pos.x, self.camera.pos.y, self.camera.pos.z, float(FIREFLY_COUNT),
            sun.x, sun.y, sun.z, 40.0)
        self.frame_ubo.update(bytes(glm.transpose(view_proj))
                              + bytes(glm.transpose(light_vp)) + tail)

    def record(self, cmd, view_proj, day, sun_uv, sun_vis, dt, now):
        cmd.begin()
        timers = {}

        with cmd.label("cull"):
            with cmd.timer() as timers["cull"]:
                cmd.fill_buffer(self.scene.visible_counter, 0)
                (cmd.bind_pipeline(self.cull_pipe)
                    .bind_descriptor_set(self.cull_set, self.cull_pipe)
                    .push_constants(self.cull_pipe, offset=0,
                                    data=bytes(glm.transpose(view_proj))
                                    + struct.pack("<I", self.scene.submesh_count))
                    .dispatch((self.scene.submesh_count + 63) // 64))

        with cmd.label("fireflies"):
            self.fireflies.record_simulation(cmd, dt, now)

        with cmd.label("shadow"):
            with cmd.timer() as timers["shadow"]:
                self.shadows.record(cmd, self.bindless_set)

        with cmd.label("scene"):
            with cmd.timer() as timers["scene"]:
                # ONE rendering scope: an MSAA target refuses
                # clear_color=None, so sky, geometry, glass and fireflies are
                # pipeline switches, not passes.
                with cmd.rendering(self.scene_rt, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
                    (c.bind_pipeline(self.sky_pipe)
                      .bind_descriptor_set(self.sky_set, self.sky_pipe)
                      .push_constants(self.sky_pipe, offset=0,
                                      data=self.camera.sky_push())
                      .draw(3))
                    (c.bind_pipeline(self.scene_pipe)
                      .bind_descriptor_set(self.scene_set, self.scene_pipe, set=0)
                      .bind_descriptor_set(self.bindless_set, self.scene_pipe, set=1))
                    (self.scene.bind_geometry(c)
                      .draw_indexed_indirect(self.scene.cull_args,
                                             count=self.scene.submesh_count))
                    if self.scene.glass_count:
                        (c.bind_pipeline(self.glass_pipe)
                          .bind_descriptor_set(self.scene_set, self.glass_pipe, set=0)
                          .bind_descriptor_set(self.bindless_set, self.glass_pipe, set=1)
                          .draw_indexed_indirect(self.scene.glass_args,
                                                 count=self.scene.glass_count))
                    self.fireflies.record_draw(c)

        with cmd.label("post"):
            with cmd.timer() as timers["post"]:
                self.post.record(cmd, day, sun_uv, sun_vis)

        with cmd.label("composite"):
            self.post.record_composite(cmd, day)

        return timers

    def read_slot_timers(self, slot):
        """Fold the timers recorded frames_in_flight ago into smoothed times.

        Read BEFORE this slot's command buffer records again — after that,
        Timer.ms raises StateError by design.
        """
        timers = self.slot_timers[slot]
        if not timers or not self.timing_ok:
            return
        try:
            for name, timer in timers.items():
                ms = timer.ms
                if ms is not None:
                    self.pass_ms[name] = self.pass_ms.get(name, ms) * 0.9 + ms * 0.1
        except bz.UnsupportedError:
            self.timing_ok = False  # this GPU never answers; stop asking
        except bz.StateError:
            pass

    def update_title(self, fps):
        self.last_visible = int(self.scene.visible_counter.read(np.uint32)[0])
        hour, minute = int(self.day.hour), int(self.day.hour % 1.0 * 60)
        parts = [f"Bazalt Demo - Showcase | {hour:02d}:{minute:02d}",
                 f"{self.last_visible}/{self.scene.submesh_count} draws",
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

    def run(self):
        print(__doc__)
        last_time = start = time.time()
        frame_index = frames = 0
        fps_timer = 0.0

        while self.window.is_open():
            bz.poll_events()
            if self.window.is_key_pressed(bz.Key.ESCAPE):
                break
            if self.window.was_key_pressed(bz.Key.F):
                self.mode_index = (self.mode_index + 1) % len(WINDOW_MODES)
                self.window.set_mode(WINDOW_MODES[self.mode_index])

            self.ctx.begin_frame()
            if not self.renderer.acquire():
                continue

            now_wall = time.time()
            dt = min(now_wall - last_time, 0.1)
            last_time = now_wall
            now = now_wall - start

            self.day.update(self.window, dt)
            self.camera.update(self.window, dt)

            day = self.day.state(now)
            view_proj = self.camera.view_proj()
            light_vp = self.shadows.light_matrix(self.camera, day["light_dir"])
            sun_uv, sun_vis = self.day.sun_screen(view_proj, self.camera, day["sun"])
            self.update_frame_ubo(view_proj, light_vp, day, now)

            slot = frame_index % self.slot_count
            self.read_slot_timers(slot)
            cmd = self.cmds[slot]
            self.slot_timers[slot] = self.record(cmd, view_proj, day, sun_uv,
                                                 sun_vis, dt, now)
            self.renderer.present(cmd)
            frame_index += 1

            frames += 1
            fps_timer += dt
            if fps_timer >= 1.0:
                self.update_title(frames / fps_timer)
                frames = 0
                fps_timer = 0.0


if __name__ == "__main__":
    DemoApp().run()
