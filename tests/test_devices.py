"""0.14: choosing the GPU.

list_devices() answers "what is on this machine" before any Context exists —
which is the only moment the answer is useful, since the answer decides which
Context to build. A Device is inert data (Device.hpp explains why it cannot
hold a VkPhysicalDevice), so these tests also pin that it survives independently
of any Context.
"""

import pytest

import bazalt as bz

# Read off the enum rather than listed by hand: a hand-written list went stale
# for four releases running, and this one covers a row the day it is added. It
# also makes the parametrized test below the referee for the 0.21 pNext column —
# a Feature whose table row sets no member pointer answers False forever, and
# nothing else would notice.
ALL_FEATURES = list(bz.Feature.__members__.values())


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
    # Both sides read the same feature table through the same query helper, and
    # this is what keeps them from drifting apart again. The Context answers
    # "enabled", the Device answers "available", so the Device may say True where
    # the Context says False — but never the other way round.
    for feature in ALL_FEATURES:
        if ctx.supports(feature):
            assert chosen[0].supports(feature), (
                f"the Context enabled {feature!r} on a device that reports it missing"
            )
