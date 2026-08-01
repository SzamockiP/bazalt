"""Two live Contexts in one process (0.15).

Until 0.15 a second Context was refused outright: volk installs its function
pointers as process globals and volkLoadDevice binds them to one VkDevice, so
the second Context would have silently redirected the first one's GPU calls at
its own device. Per-Context dispatch tables (volkLoadDeviceTable) removed that.

What is actually under test is dispatch crosstalk between two VkDevices, which
happens on ONE GPU just as readily as on two — so all of this runs anywhere,
lavapipe included. `list_devices()` picks a second physical GPU when the machine
has one; when it doesn't, the same device twice is still two devices, two
instances and two dispatch tables.

Every Context here is local to its test: the session-scoped `ctx` fixture stays
session-scoped for the reason it always had (every resource keeps its Context
alive, so per-test Contexts are a leak trap), not because of volk.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


@pytest.fixture
def pair(extra_context):
    """Two Contexts, on two GPUs where the machine has them.

    Both go through extra_context, so the validation layers referee them the
    same way they referee the session Context."""
    devices = bz.list_devices()
    return (extra_context(),
            extra_context(device=devices[1] if len(devices) > 1 else None))


def render_flat(context, color):
    """A cleared 16x16 target, read back. Enough GPU traffic to catch a command
    recorded through the wrong device's dispatch table."""
    target = context.create_render_target(16, 16)
    cmd = context.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=color)
    cmd.end_rendering(target)
    context.submit(cmd)
    return target.color[0].read()


def test_two_contexts_are_alive_at_once(pair):
    a, b = pair
    assert a.device_name and b.device_name


def test_each_context_renders_its_own_work(pair):
    a, b = pair
    assert render_flat(a, [1.0, 0.0, 0.0, 1.0])[0, 0, 0] == 255
    assert render_flat(b, [0.0, 0.0, 1.0, 1.0])[0, 0, 2] == 255


def test_interleaved_submits_stay_independent(pair):
    """A/B/A/B in one loop. Each Context has its own frame ring, submission
    timeline and deletion queue; interleaving must not braid them."""
    a, b = pair
    for i in range(3):
        a.begin_frame()
        b.begin_frame()
        assert render_flat(a, [1.0, 0.0, 0.0, 1.0])[0, 0, 0] == 255
        assert render_flat(b, [0.0, 1.0, 0.0, 1.0])[0, 0, 1] == 255
    assert a.frame_index is not None and b.frame_index is not None


def test_contexts_can_differ_in_validation_settings(extra_context):
    """Validation layers are per-instance, so two Contexts must be able to
    disagree about them — this is what lets the sync-validation tests run
    alongside the session Context instead of in a subprocess."""
    quiet = extra_context(validation="off")
    loud = extra_context(validation="on")
    assert render_flat(quiet, [1.0, 1.0, 1.0, 1.0])[0, 0, 0] == 255
    assert render_flat(loud, [1.0, 1.0, 1.0, 1.0])[0, 0, 0] == 255


def test_one_context_outlives_the_other(pair):
    """Destroying B must not disturb A: nothing device-level is shared, and the
    globals A relies on are instance-level loader trampolines."""
    a, b = pair
    del b
    assert render_flat(a, [1.0, 0.0, 0.0, 1.0])[0, 0, 0] == 255


def test_every_listed_device_can_back_a_context(extra_context):
    for device in bz.list_devices():
        context = extra_context(device=device)
        assert context.device_name == device.name


# ── Resources do not cross Contexts ────────────────────────────────────────────


def test_foreign_image_in_a_command_buffer_is_refused(pair):
    a, b = pair
    image = a.create_image(np.zeros((4, 4, 4), dtype=np.uint8))
    cmd = b.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError) as info:
        cmd.barrier(image, bz.Access.SHADER_READ, bz.Access.SHADER_READ)
    assert "different Context" in str(info.value)


def test_foreign_pipeline_in_a_command_buffer_is_refused(pair):
    a, b = pair
    pipeline = (a.graphics_pipeline()
                .vertex_shader(a.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX))
                .fragment_shader(a.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT))
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(a.create_render_target(16, 16)))
    cmd = b.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError):
        cmd.bind_pipeline(pipeline)


def test_foreign_target_in_begin_rendering_is_refused(pair):
    a, b = pair
    target = a.create_render_target(16, 16)
    cmd = b.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError):
        cmd.begin_rendering(target)


def test_foreign_buffer_in_a_descriptor_set_is_refused(pair):
    a, b = pair
    buffer = a.create_buffer([1.0, 2.0, 3.0, 4.0], bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    shader = b.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    pipeline = b.compute_pipeline().shader(shader).storage_buffer(0).build()
    pool = b.create_descriptor_pool(max_sets=1, storage_buffers=1)
    dset = pool.allocate_set(pipeline, set=0)
    with pytest.raises(bz.ResourceError):
        dset.set_buffer(0, buffer)


# ── Moving pixels between Contexts ─────────────────────────────────────────────


def test_transferred_image_matches_the_source(pair):
    a, b = pair
    pixels = np.arange(8 * 8 * 4, dtype=np.uint8).reshape(8, 8, 4)
    source = a.create_image(pixels)
    copy = b.create_image(source)
    copy.wait()

    assert np.array_equal(copy.read(), pixels)
    assert (copy.width, copy.height) == (8, 8)
    assert copy.format == source.format


def test_transferred_image_is_usable_on_the_target_context(pair):
    """The copy must be a first-class Image of B — bindable in B's recordings,
    which is the whole point of moving it."""
    a, b = pair
    source = a.create_image(np.full((4, 4, 4), 200, dtype=np.uint8))
    copy = b.create_image(source)

    cmd = b.create_command_buffer()
    cmd.begin()
    cmd.barrier(copy, bz.Access.SHADER_READ, bz.Access.SHADER_READ)
    b.submit(cmd)


def test_transfer_carries_layers_and_cubeness(pair):
    """What the numpy round trip cannot do: image.read() returns mip 0 of layer 0,
    so `b.create_image(a_cube.read())` would silently flatten a cubemap."""
    a, b = pair
    faces = [np.full((4, 4, 4), i * 40, dtype=np.uint8) for i in range(6)]
    source = a.create_image(faces, cube=True)
    copy = b.create_image(source)
    copy.wait()

    assert copy.is_cube
    assert copy.array_layers == 6
    assert np.array_equal(copy.read(), faces[0])


def test_transfer_regenerates_the_mip_chain(pair):
    """A mipped source arrives mipped. The levels are regenerated on the target,
    not copied — a documented ceiling, not an accident."""
    a, b = pair
    source = a.create_image(np.full((8, 8, 4), 128, dtype=np.uint8), mipmaps=True)
    copy = b.create_image(source)
    copy.wait()
    assert copy.mip_levels == source.mip_levels > 1


def test_transfer_within_one_context_is_a_clone(ctx):
    """Source and destination being the same Context is legal (and useful)."""
    pixels = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    source = ctx.create_image(pixels)
    clone = ctx.create_image(source)
    clone.wait()
    assert np.array_equal(clone.read(), pixels)


def test_transfer_of_an_empty_image_is_refused(pair):
    a, b = pair
    empty = a.create_image(4, 4)
    with pytest.raises(bz.ResourceError):
        b.create_image(empty)


# ── 0.18: the mip chain crosses too ──


def test_a_transferred_image_carries_its_own_mip_levels(extra_context):
    """0.15 regenerated levels 1..N on the far side and recorded the loss as a
    ceiling: "a hand-authored mip chain flattens into a generated one". That is
    a silent wrong answer for anyone who rendered their own levels — a
    roughness-prefiltered environment map is exactly that.

    Written so the generated chain and the authored one cannot agree: mip 0 is
    red and mip 1 is green, which no filter of red produces.
    """
    a = extra_context()
    b = extra_context()

    red = np.zeros((16, 16, 4), dtype=np.uint8)
    red[:, :, 0] = 255
    red[:, :, 3] = 255
    green = np.zeros((8, 8, 4), dtype=np.uint8)
    green[:, :, 1] = 255
    green[:, :, 3] = 255

    source = a.create_image(red, mipmaps=True)
    source.update(green, mip=1)
    source.wait()
    assert source.read(mip=1)[0, 0].tolist() == [0, 255, 0, 255]

    copy = b.create_image(source)
    copy.wait()

    assert copy.read(mip=0)[0, 0].tolist() == [255, 0, 0, 255]
    assert copy.read(mip=1)[0, 0].tolist() == [0, 255, 0, 255], \
        "mip 1 was regenerated from mip 0 instead of copied"
