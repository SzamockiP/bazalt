"""0.14: choosing the GPU.

list_devices() answers "what is on this machine" before any Context exists —
which is the only moment the answer is useful, since the answer decides which
Context to build. A Device is inert data (Device.hpp explains why it cannot
hold a VkPhysicalDevice), so these tests also pin that it survives independently
of any Context.
"""

import subprocess
import sys

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
        assert d.limits.device_memory >= 0


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


def test_list_devices_works_as_the_first_call(tmp_path):
    """In a subprocess, because the bug only exists when nothing ran first.

    query_device_features reads the device extension list, and until 0.26 it
    read it through volk's instance-level global — which is null until a
    Context calls volkLoadInstanceOnly. Every test in this file passes either
    way, because the session Context is already up by the time they run. A
    PROGRAM that opens with bz.list_devices(), which is what the function is
    for, segfaulted on its first line.
    """
    script = tmp_path / "first_call.py"
    script.write_text(
        "import bazalt as bz\n"
        "devices = bz.list_devices()\n"
        "print(len(devices))\n"
        # Touch the part that needed the extension list, so a crash-free but
        # empty answer does not pass either.
        "for d in devices:\n"
        "    d.supports(bz.Feature.EXCLUSIVE_FULLSCREEN)\n",
        encoding="utf-8")

    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, f"exit {done.returncode}\nstdout: {done.stdout}\nstderr: {done.stderr}"


def test_a_device_answers_the_same_limits_a_context_does(ctx):
    """One set of numbers about a GPU, reachable from either side.

    `Device.memory_mb` used to sit beside `ctx.limits`, so what a caller asked
    depended on whether a Context existed yet. It is `limits.device_memory` now
    and both paths fill it from the same heap rule — a difference here would
    mean the two had drifted, which is the thing having one type prevents.
    """
    same = [d for d in bz.list_devices() if d.name == ctx.device_name]
    if not same:
        pytest.skip("the Context is not on a device list_devices reports")

    device = same[0]
    assert device.limits.device_memory == ctx.limits.device_memory
    assert device.limits.max_storage_buffer == ctx.limits.max_storage_buffer
    assert device.limits.max_workgroup_size == ctx.limits.max_workgroup_size


def test_the_memory_budget_stays_within_the_device(ctx):
    """memory_stats() counts the device-local heaps and nothing else.

    It summed every heap until 0.26, which on a laptop includes the system
    memory the GPU may spill into — an 8 GiB card reported a budget of 18.9
    GiB. Code that sizes a load against that number fills VRAM and then crawls.
    """
    stats = ctx.memory_stats()
    capacity = ctx.limits.device_memory
    assert stats.budget <= capacity, (
        f"budget {stats.budget} exceeds the device-local heaps ({capacity}), "
        "so it is counting host memory again")
    assert stats.used <= stats.reserved
