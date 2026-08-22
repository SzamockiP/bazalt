"""Async compute: a pass on the second queue (0.29).

Every pass names its queue. Since 0.29 `bz.Queue` has a second member:

    graph.add_pass(name="produce", queue=bz.Queue.COMPUTE)

That pass runs on the device's compute queue, beside the graphics work rather
than through it. What the graph then arranges for you is worth stating exactly,
because it is two things and not three:

1. Inside one graph, a pass that reads what a pass on the OTHER queue wrote is
   ordered against it. A pipeline barrier cannot reach across a queue, so the
   graph turns that edge into a timeline-semaphore wait between the submits it
   makes.
2. A graph you submit AGAIN waits for its own previous submit on the other
   queue. Replay N+1's producer would otherwise overwrite the numbers replay
   N's consumer is still reading.

And what it does not arrange: between two DIFFERENT graphs on different queues
there is no order at all. That is deliberate — two graphs are how you say "these
may overlap" — and `ctx.submit(g, after=serial)` is how you order them when you
mean to. See examples/44_submit_order for that half.

The chain here is a producer and a consumer over one buffer, replayed N times.
The consumer sums what the producer wrote, so the answer is 1+2+...+N, and any
missing edge — inside a replay or between two of them — gives a smaller number.
Running it with the producer on each queue in turn must give the same number
twice.

A device without a compute-only queue family still runs all of this: the compute
queue aliases the graphics one and keeps its own timeline, so the program and its
ordering are unchanged and only the overlap is missing. The line below reports
which of the two you have.

Sync validation is on, because it is the mode that reports a MISSING barrier,
and a cross-queue dependency is exactly the kind nothing else would notice.

Run it with no arguments.
"""

import time

import numpy as np

import bazalt as bz

N = 4096
GROUPS = N // 64
STEPS = 24

hazards = []
logger = bz.Logger(min_severity=bz.Severity.INFO)


@logger.on_message
def on_message(msg):
    if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
        hazards.append(msg.text)
    elif msg.severity >= bz.Severity.WARNING:
        print(f"[{msg.severity}] {msg.text}")


ctx = bz.Context(logger, validation="sync")

produce = (ctx.compute_pipeline()
           .shader(ctx.compile_shader("produce.comp", bz.ShaderStage.COMPUTE))
           .storage_buffer(0)
           .build())

accumulate = (ctx.compute_pipeline()
              .shader(ctx.compile_shader("accumulate.comp", bz.ShaderStage.COMPUTE))
              .storage_buffer(0)
              .storage_buffer(1)
              .build())


def run(queue):
    """Replay one producer/consumer graph STEPS times, the producer on `queue`.

    Nothing waits on the CPU until the end: every submit is asynchronous, so
    the queues really do have several replays in flight and the ordering under
    test is the GPU's rather than Python's.
    """
    source = ctx.create_buffer(np.zeros(N, dtype=np.uint32),
                              bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    destination = ctx.create_buffer(np.zeros(N, dtype=np.uint32),
                                    bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    pool = ctx.create_descriptor_pool()
    produce_set = pool.allocate_set(produce)
    produce_set.set_buffer(0, source)
    accumulate_set = pool.allocate_set(accumulate)
    accumulate_set.set_buffer(0, source)
    accumulate_set.set_buffer(1, destination)

    graph = ctx.graph()
    (graph.add_pass(name="produce", queue=queue)
        .bind_pipeline(produce)
        .bind_descriptor_set(produce_set, produce)
        .dispatch(GROUPS))
    (graph.add_pass(name="accumulate")
        .bind_pipeline(accumulate)
        .bind_descriptor_set(accumulate_set, accumulate)
        .dispatch(GROUPS))

    start = time.perf_counter()
    for _ in range(STEPS):
        ctx.submit(graph, wait=False)
    ctx.wait()
    elapsed = (time.perf_counter() - start) * 1000.0

    return int(destination.read(np.uint32)[0]), elapsed


separate = ctx.supports(bz.Feature.ASYNC_COMPUTE)
print(f"device: {ctx.device_name}")
print(f"separate compute queue family: {'yes' if separate else 'no, the compute queue is the graphics one'}")

expected = STEPS * (STEPS + 1) // 2
results = {}
for name, queue in (("graphics", bz.Queue.GRAPHICS), ("compute", bz.Queue.COMPUTE)):
    value, ms = run(queue)
    results[name] = value
    print(f"producer on {name:8} -> {value:6}  ({ms:6.2f} ms for {STEPS} replays)")

# The milliseconds are wall clock for a deliberately tiny workload, so submit
# overhead dominates them and the two numbers trade places between runs. They
# are here to show the program did something, not to measure the overlap: for
# that the compute pass has to be long enough to hide behind the graphics work,
# which a demo that must finish in a second cannot be.

logger.flush()
print(f"expected {expected}")
print(f"sync-validation hazards: {len(hazards)}")
for text in hazards[:3]:
    print(f"  {text}")

assert results["graphics"] == expected, "the single-queue chain lost an edge"
assert results["compute"] == expected, "a cross-queue edge is missing"
assert not hazards, "sync validation reported a missing barrier"
print("\nBoth queues agree, and the layers are quiet.")
