"""0.14: choosing the GPU.

list_devices() answers "what is on this machine" before any Context exists —
which is the only moment the answer is useful, since the answer decides which
Context to build. A Device is inert data (Device.hpp explains why it cannot
hold a VkPhysicalDevice), so these tests also pin that it survives independently
of any Context.
"""

import pytest

import bazalt as bz

ALL_FEATURES = [
    bz.Feature.ANISOTROPIC_FILTERING,
    bz.Feature.WIREFRAME,
    bz.Feature.WIDE_LINES,
    bz.Feature.DEPTH_CLAMP,
    bz.Feature.SAMPLE_RATE_SHADING,
    bz.Feature.MULTI_DRAW_INDIRECT,
    bz.Feature.SHADER_FLOAT64,
]


def test_list_devices_finds_at_least_one_gpu():
    """The session Context exists, so a GPU demonstrably does too."""
    devices = bz.list_devices()
    assert devices, "a machine running these tests has at least one Vulkan device"


def test_list_devices_needs_no_context():
    """It builds and destroys its own instance, so it must not disturb — or
    depend on — the live session Context. Called twice to catch a listing that
    only works the first time."""
    first = bz.list_devices()
    second = bz.list_devices()
    assert [d.name for d in first] == [d.name for d in second]


def test_device_reports_readable_properties():
    for d in bz.list_devices():
        assert d.name
        assert d.type in ("discrete", "integrated", "virtual", "cpu", "other")
        assert d.api_version.count(".") == 2
        assert d.memory_mb >= 0
        assert isinstance(d.supports_multiview(), bool)


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_device_answers_supports_for_every_feature(feature):
    for d in bz.list_devices():
        assert isinstance(d.supports(feature), bool)


def test_the_automatic_choice_is_one_of_the_listed_devices(ctx):
    """The default path and the listing must agree about what exists —
    otherwise `Context(device=...)` would be picking from a different set than
    the one the user was shown."""
    assert ctx.device_name in [d.name for d in bz.list_devices()]


def test_devices_outlive_the_listing_instance():
    """A Device holds copies, not handles: reading one after the instance that
    enumerated it is long gone must be safe, not a use-after-free."""
    devices = bz.list_devices()
    for _ in range(3):
        bz.list_devices()
    assert all(d.name and d.type for d in devices)


def test_repr_names_the_card():
    d = bz.list_devices()[0]
    assert d.name in repr(d)


def test_explicit_device_is_honoured(ctx):
    """Only one Context may live per process, so this cannot build a second one
    to check. What it can check is that the device the automatic path chose is
    addressable by handle — which is what Context(device=...) consumes."""
    chosen = [d for d in bz.list_devices() if d.name == ctx.device_name]
    assert chosen, "the running Context's GPU must appear in list_devices()"
    assert chosen[0].supports_multiview() == ctx.supports_multiview()
