"""Reading a buffer through its address, past what a descriptor can see.

A descriptor may only reach `ctx.limits.max_storage_buffer` bytes of a buffer.
On the desktop drivers that report the most that is 4 GiB minus one byte, and it
is a property of the BINDING, not of the memory: `ctx.limits.max_buffer` is
usually far larger. Feature.BUFFER_ADDRESS is how to spend the difference — the
shader takes a pointer in a push constant and no descriptor is involved at all.

Three things at once, because they are the same job:

  * `ctx.limits` — the numbers that say what this GPU allows;
  * `buffer.address` in a push constant, read in the shader as a
    `buffer_reference` pointer;
  * `layout(local_size_x_id = 0)` — a workgroup size this file picks and the
    pipeline bakes in, sized against `limits.max_workgroup_invocations`.

Headless on purpose: nothing here is a picture.

    python main.py            # 256 MiB
    python main.py --mib 8192 # as much as the GPU will give
"""

import argparse
import struct
import time

import numpy as np

import bazalt as bz


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mib", type=int, default=256, help="how much to allocate, in MiB")
    args = parser.parse_args()

    ctx = bz.Context(features=[bz.Feature.BUFFER_ADDRESS, bz.Feature.SHADER_INT64],
                     optional=[bz.Feature.WORKGROUP_SIZE])
    print(f"{ctx.device_name}, Vulkan {ctx.api_version}")

    limits = ctx.limits
    print(f"one descriptor reaches   {limits.max_storage_buffer / 2**30:8.2f} GiB")
    print(f"one buffer may be        {limits.max_buffer / 2**30:8.2f} GiB")
    print(f"one allocation may be    {limits.max_allocation / 2**30:8.2f} GiB")
    print(f"workgroup                {limits.max_workgroup_size}, {limits.max_workgroup_invocations} threads")
    print(f"subgroup                 {ctx.subgroup_size} ({limits.min_subgroup_size}..{limits.max_subgroup_size})")

    # A size the device says it will take, rather than a number typed here.
    group = min(256, limits.max_workgroup_invocations)
    if not ctx.supports(bz.Feature.WORKGROUP_SIZE):
        # Without maintenance4 the shader's own local_size stands, and the
        # spec constant is ignored — so match what the text says.
        print("no maintenance4: the workgroup size in the shader text wins")
        group = 64

    words = args.mib * 2**20 // 4
    print(f"\nfilling {args.mib} MiB ({words:,} words)...")

    started = time.perf_counter()
    data = np.arange(words, dtype=np.uint32)
    # Staged in 64 MiB pieces, so this needs 64 MiB of host memory rather than
    # another {args.mib}. That is what lets the number above go up.
    big = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC, name="big")
    out = ctx.create_buffer(
        np.zeros(group, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC, name="out")
    print(f"uploaded in {time.perf_counter() - started:.2f} s")

    pipeline = (ctx.compute_pipeline()
                .shader(ctx.compile_shader("sum.comp", bz.ShaderStage.COMPUTE))
                .push_constant(24)
                .constant(0, group)
                .name("sum")
                .build())

    # No descriptor set: both buffers travel as addresses. Reading .address
    # waits for the upload above, because nothing else can see the dependency.
    push = struct.pack("<QQQ", big.address, out.address, words)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.timer() as gpu:
        cmd.bind_pipeline(pipeline).push_constants(pipeline, 0, push).dispatch(1)
    ctx.submit(cmd)

    # Every partial sum wrapped at 2^32 in the shader, so the total only agrees
    # with the real one modulo that.
    total = int(out.read(np.uint32)[:group].sum()) % 2**32
    expected = (words * (words - 1) // 2) % 2**32
    print(f"\nsum mod 2^32: got {total}, expected {expected}")
    assert total == expected, "the shader did not read what was uploaded"

    try:
        rate = (args.mib / 1024) / (gpu.ms / 1000)
        print(f"read {args.mib} MiB in {gpu.ms:.2f} ms — {rate:.0f} GiB/s")
    except bz.UnsupportedError:
        print("this device reports no usable timestamps")


if __name__ == "__main__":
    main()
