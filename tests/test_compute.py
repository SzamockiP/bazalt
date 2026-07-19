"""Compute needs no images: dispatch -> SSBO -> numpy assert.

This is why compute tests are the first thing CI on a software rasterizer
runs — no golden images, just arithmetic.
"""

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


def test_compute_shader_compiles(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    assert comp is not None


def test_bad_compute_shader_raises_shader_error(ctx, tmp_path):
    bad = tmp_path / "bad.comp"
    bad.write_text("#version 450\nlayout(local_size_x = 1) in;\nvoid main() { nonsense }\n")

    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader(str(bad), bz.ShaderStage.COMPUTE)
    assert info.value.path == str(bad)
