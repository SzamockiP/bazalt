"""Ordering whole submits by hand: Serial and after= (0.28).

Inside one graph the passes order themselves — the graph reads what each pass
touches and puts the barriers in. Between SUBMITS there is nothing to read,
so 0.28 gives you the handle instead:

    serial = ctx.submit(graph, wait=False)      # returns a Serial
    ctx.submit(other, wait=False, after=serial) # starts after that one
    ctx.wait(serial)                            # wait for that one alone

The two words answer different questions and both exist for it:

  * `wait=` is about the CPU. wait=True blocks until the GPU is done, which
    is right when the next line reads the result.
  * `after=` is about the GPU. The submit waits on the device for the submits
    you name, and the CPU carries on immediately.

A `Serial` is opaque on purpose: no attributes, no ordering, no arithmetic.
It names a submit, and nothing else. That is what lets async compute add a
second queue later without any code here changing meaning — two serials from
two queues were never comparable, so nothing can have relied on it.

This example runs headless and checks itself. It runs the same chain three
ways and prints the wall-clock cost of each:

  1. blocking, one submit at a time — the CPU waits for every step;
  2. asynchronous with after=, chained — same order on the GPU, no CPU waits;
  3. asynchronous with no ordering at all — which is why after= exists. Do
     not copy this one: nothing tells the GPU those submits depend on each
     other, so the answer is undefined. It may well come out right on your
     machine, and that proves nothing. The ring paces submits when the frames
     in flight run out, which hides the problem some of the time, and a
     driver is free to run them in parallel whenever it can.

Since 0.29 there is a second queue, and `after=` is also how work on one queue
waits for work on the other. The spelling here does not change.
examples/45_async_compute shows that half.

Run it with no arguments.
"""

import struct
import time

import numpy as np

import bazalt as bz

N = 1 << 16
GROUPS = N // 64
STEPS = 8

logger = bz.Logger()


@logger.on_message
def on_message(msg):
    if msg.severity >= bz.Severity.ERROR:
        print(f"[{msg.severity}] {msg.text}")


ctx = bz.Context(logger)

advance = (ctx.compute_pipeline()
           .shader(ctx.compile_shader("advance.comp", bz.ShaderStage.COMPUTE))
           .storage_buffer(0)
           .push_constant(4)
           .build())

pool = ctx.create_descriptor_pool()


def make_step_graphs(values):
    """One graph per step. Each is a submit of its own, which is the point:
    the ordering under test is BETWEEN submits, not inside one."""
    dset = pool.allocate_set(advance)
    dset.set_buffer(0, values)
    graphs = []
    for step in range(1, STEPS + 1):
        g = ctx.graph()
        (g.add_pass(name=f"step {step}")
            .bind_pipeline(advance)
            .bind_descriptor_set(dset, advance)
            .push_constants(advance, 0, struct.pack("<I", step))
            .dispatch(GROUPS))
        graphs.append(g)
    return graphs


def expected_value():
    """What the chain computes when the steps really run in order."""
    v = 0
    for step in range(1, STEPS + 1):
        v = v * 2 + step
    return v


def run(mode):
    values = ctx.create_buffer(np.zeros(N, dtype=np.uint32),
                               bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    graphs = make_step_graphs(values)

    start = time.perf_counter()
    if mode == "blocking":
        for g in graphs:
            # The default. Every submit round-trips to the GPU and back.
            ctx.submit(g)
    elif mode == "after":
        previous = None
        for g in graphs:
            # No CPU wait anywhere in this loop. Each submit carries the
            # dependency on the one before it, and the GPU honours it.
            previous = ctx.submit(g, wait=False, after=previous)
        # Wait for the LAST serial only — everything before it is a
        # prerequisite of that one, so this covers the chain.
        ctx.wait(previous)
    else:
        for g in graphs:
            # Deliberately wrong, and here to show what after= buys: nothing
            # says these must not overlap.
            ctx.submit(g, wait=False)
        ctx.wait()
    elapsed = (time.perf_counter() - start) * 1000.0

    return elapsed, int(values.read(np.uint32)[0])


want = expected_value()
print(f"{STEPS} steps over {N} elements, expected {want}\n")
print(f"{'mode':<28}{'ms':>8}   result")

results = {}
for mode, label in (("blocking", "wait=True per submit"),
                    ("after", "wait=False + after="),
                    ("unordered", "wait=False, unordered (unsafe)")):
    ms, value = run(mode)
    results[mode] = value
    ok = "ok" if value == want else f"WRONG (got {value})"
    print(f"{label:<28}{ms:>8.2f}   {ok}")

print()
assert results["blocking"] == want, "the blocking chain is wrong, which is a bug"
assert results["after"] == want, "after= did not order the submits"
if results["unordered"] != want:
    print("The unordered run is wrong, which is the expected outcome: nothing")
    print("told the GPU those submits depend on each other.")
else:
    print("The unordered run happened to come out right on this GPU. It is not")
    print("ordered, so it is not guaranteed to — that is what after= is for.")
