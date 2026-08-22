"""The escape hatch: one manual pass beside the automatic ones (0.28).

Bazalt computes the barriers, and rule 2 of the design says it must never
become a ceiling — there has to be a way to do it yourself. In 0.28 that way
is a PASS rather than a second API:

    graph.add_pass(auto_barriers=False)

Only that pass goes manual. Its hazards are yours to express with
`p.barrier()`, and the passes around it keep their automatic barriers. The
two compose because the graph reads the barriers you wrote: a manual pass
tells the fold what state it left the buffer in, so the automatic pass after
it orders against that and does not stack a second barrier on top.

That composition is the whole point, and it is why the first design of this
hatch was rejected. A raw recorder BESIDE the graph would have meant "manual
for this part" was really "manual for the whole frame".

This example runs headless and checks itself, so it is also a small proof:
the same chain is built twice, once fully automatic and once with the middle
pass manual, and the two must produce the same number. Sync validation is the
referee — it is the mode that reports a MISSING barrier, which core
validation cannot see.

Run it with no arguments.
"""

import struct

import numpy as np

import bazalt as bz

N = 4096
GROUPS = N // 64

hazards = []
logger = bz.Logger(min_severity=bz.Severity.INFO)


@logger.on_message
def on_message(msg):
    if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
        hazards.append(msg.text)
    elif msg.severity >= bz.Severity.WARNING:
        print(f"[{msg.severity}] {msg.text}")


# validation="sync" costs real time and is a debugging mode, not a default.
# It is on here because this example is about barriers, and it is the only
# mode that reports one that is missing.
ctx = bz.Context(logger, validation="sync")

scale = (ctx.compute_pipeline()
         .shader(ctx.compile_shader("scale.comp", bz.ShaderStage.COMPUTE))
         .storage_buffer(0)
         .push_constant(4)
         .build())

total = (ctx.compute_pipeline()
         .shader(ctx.compile_shader("total.comp", bz.ShaderStage.COMPUTE))
         .storage_buffer(0)
         .storage_buffer(1)
         .build())


def run(manual_middle):
    """Scale twice, then sum. The middle pass goes manual when asked.

    Same work either way. The only difference is who writes the barrier
    between the second scaling and everything around it.
    """
    values = ctx.create_buffer(np.ones(N, dtype=np.float32),
                               bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    accumulator = ctx.create_buffer(np.zeros(1, dtype=np.uint32),
                                    bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    pool = ctx.create_descriptor_pool()
    scale_set = pool.allocate_set(scale)
    scale_set.set_buffer(0, values)
    total_set = pool.allocate_set(total)
    total_set.set_buffer(0, values)
    total_set.set_buffer(1, accumulator)

    graph = ctx.graph()

    # Automatic: the graph sees the write and orders whatever reads next.
    (graph.add_pass(name="scale by 3")
        .bind_pipeline(scale)
        .bind_descriptor_set(scale_set, scale)
        .push_constants(scale, 0, struct.pack("<f", 3.0))
        .dispatch(GROUPS))

    # The middle pass. With auto_barriers=False nothing is computed for it,
    # so the write-after-write against the pass above is spelled by hand.
    middle = graph.add_pass(
        name="scale by 2 (manual)" if manual_middle else "scale by 2",
        auto_barriers=False if manual_middle else None)
    if manual_middle:
        # WAW: this dispatch overwrites what the pass above just wrote.
        middle.barrier(values, bz.Access.SHADER_WRITE, bz.Access.SHADER_WRITE)
    (middle.bind_pipeline(scale)
        .bind_descriptor_set(scale_set, scale)
        .push_constants(scale, 0, struct.pack("<f", 2.0))
        .dispatch(GROUPS))
    if manual_middle:
        # RAW: and the pass below reads it. This is also what keeps the two
        # halves composable — the note reaches the fold, so the automatic pass
        # after this one orders against what was just declared here instead of
        # stacking a second barrier for the same dependency.
        middle.barrier(values, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)

    # Automatic again, reading what the manual pass wrote.
    (graph.add_pass(name="sum")
        .bind_pipeline(total)
        .bind_descriptor_set(total_set, total)
        .dispatch(GROUPS))

    ctx.submit(graph)
    return int(accumulator.read(np.uint32)[0])


automatic = run(manual_middle=False)
manual = run(manual_middle=True)
logger.flush()

expected = N * 6 * 1000  # ones, times 3, times 2, times the shader's 1000
print(f"expected            {expected}")
print(f"all passes automatic {automatic}")
print(f"middle pass manual   {manual}")
print(f"sync-validation hazards: {len(hazards)}")
for text in hazards[:3]:
    print(f"  {text}")

assert automatic == expected, "the automatic chain lost a barrier"
assert manual == expected, "the manual pass did not express its hazards"
assert not hazards, "sync validation reported a missing barrier"
print("\nBoth chains agree, and the layers are quiet.")
