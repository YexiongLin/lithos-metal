"""Measured launch scope and core/merge numerical partition agreement."""
import copy
import struct

import pytest

from monolith.backends.metal import config_for_device
from monolith.backends.metal.m6.attention import finalize, validate
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program


def fixture(draft=False, **macros):
    config = copy.deepcopy(config_for_device(12, 11, 'Apple M6'))
    config.raw['engine']['attention_launch'] = dict(workers=24, simdgroups=8)
    m = dict(D='128' if draft else '256', RBMAX='4u', CH='64u', **macros)
    if draft:
        m['DRAFT'] = '1'
    record = struct.pack('<20I', *([32, 8, 7, 0, 144] if draft else [24, 4, 8, 0, 144]), *([0] * 15))
    program = Program(
        {'core': KernelSpec('', 'gqa_decode', dict(m)), 'merge': KernelSpec('', 'gqa_merge', dict(m))},
        {'params': BufferSpec(80, record, 'params')},
        [OpSpec('core', [(9, 'params', 0)], (12, 1, 1), (384, 1, 1)),
         OpSpec('merge', [(4, 'params', 0)], (192, 1, 1), (32, 1, 1))])
    return config, program


@pytest.mark.parametrize('draft', [False, True])
def test_launch_keeps_core_merge_partition_and_is_idempotent(draft):
    config, program = fixture(draft)
    original = copy.deepcopy(program)
    finalize(program, None, {}, config, t=8, dynamic_t=True)
    assert program.ops[0].grid == (24, 1, 1)
    assert program.ops[0].threadgroup == (256, 1, 1)
    assert program.ops[1].grid == original.ops[1].grid
    assert program.ops[1].threadgroup == original.ops[1].threadgroup
    for op in program.ops:
        macros = program.kernels[op.kernel].macros
        assert macros['GQA_CHUNK_SG'] == '144u'
        assert macros['STATIC_GQA_P_N_SG'] == '192u'
    before, after = original.buffers['params'].init, program.buffers['params'].init
    assert struct.unpack_from('<I', after, 16)[0] == 192
    assert after[:16] == before[:16] and after[20:] == before[20:]
    assert set(program.kernels) == {op.kernel for op in program.ops}
    unchanged = program.to_json()
    finalize(program, None, {}, config, t=8, dynamic_t=True)
    assert program.to_json() == unchanged


@pytest.mark.parametrize('case', ['off', 'static', 'large_t', 'different_shape', 'gamma', 'rows', 'lm', 'mma', 'crew', 'custom_grid'])
def test_unmeasured_routes_remain_unchanged(case):
    config, program = fixture(draft=case == 'gamma', **({'LM_MODE': '1'} if case == 'lm' else {}))
    options = dict(t=8, dynamic_t=True)
    if case == 'off': config.raw['engine'].pop('attention_launch')
    if case == 'static': options['dynamic_t'] = False
    if case == 'large_t': options['t'] = 512
    if case == 'rows': config.attention_rows = 8
    if case == 'crew': config.threadgroups_per_core = 2
    if case == 'custom_grid': program.ops[0].grid = (24, 1, 1)
    if case in ('different_shape', 'gamma'):
        record = bytearray(program.buffers['params'].init)
        struct.pack_into('<I', record, 8 if case == 'gamma' else 0, 15)
        program.buffers['params'].init = bytes(record)
    if case == 'mma':
        program.kernels['core'].function = 'gqa_decode_mma'
        for k in program.kernels.values(): k.macros['FIXED_CHUNK'] = '1'
    expected = program.to_json()
    finalize(program, None, {}, config, **options)
    assert program.to_json() == expected


def test_recipe_is_validated_and_fingerprinted():
    config, _ = fixture()
    validate(config)
    original = config.fingerprint
    config.raw['engine']['attention_launch']['workers'] = 48
    assert config.fingerprint != original
    with pytest.raises(ValueError, match='measured'):
        validate(config)
