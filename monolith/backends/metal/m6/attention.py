"""Attention launch measured on the 12-core M6; preserve softmax partitions."""
import copy
import struct


MEASURED_LAUNCH = {'workers': 24, 'simdgroups': 8}


def validate(config):
    launch = config.raw.get('engine', {}).get('attention_launch')
    if launch is not None and launch != MEASURED_LAUNCH:
        raise ValueError('M6 attention_launch must match the measured 24-worker, 8-SIMD configuration')


def finalize(program, graph, emitted, config, *, t, dynamic_t, **options):
    launch = config.raw.get('engine', {}).get('attention_launch')
    if (not launch or config.key != 'apple11' or config.gpu_cores != 12
            or config.threadgroups_per_core != 1 or config.attention_rows != 4
            or not dynamic_t or t not in (8, 32, 128)):
        return program
    # Read all records before modifying any: a core and its merge can share one.
    selected = []
    integer = lambda k, name, default=0: int(str(k.macros.get(name, default)).rstrip('u'))
    for op in program.ops:
        kernel = program.kernels[op.kernel]
        if kernel.function not in ('gqa_decode', 'gqa_merge'):
            continue
        if (integer(kernel, 'FIXED_CHUNK') or integer(kernel, 'LM_MODE')
                or integer(kernel, 'RBMAX', 4) != 4 or integer(kernel, 'CH', 64) != 64):
            continue
        slot = 9 if kernel.function == 'gqa_decode' else 4
        name, offset = next((name, offset) for binding, name, offset in op.bindings if binding == slot)
        heads, kv, rows, _, nsg = struct.unpack_from('<5I', program.buffers[name].init, offset)
        draft = integer(kernel, 'DRAFT')
        if (heads, kv, integer(kernel, 'D'), draft) not in ((24, 4, 256, 0), (32, 8, 128, 1)):
            continue
        if nsg != 144 or (draft and rows != 7):
            continue
        if kernel.function == 'gqa_decode' and (op.grid != (12, 1, 1) or op.threadgroup != (384, 1, 1)):
            # Do not update a shared core/merge record under a custom launch.
            return program
        selected.append((op, name, offset, nsg))
    replaced = set()
    for op, name, offset, partition_sg in selected:
        # A source kernel may serve other shapes; specialize this launch without
        # changing those consumers or any projection/recurrent dispatch.
        key = op.kernel + '|m6_attention'
        replaced.add(op.kernel)
        if key not in program.kernels:
            kernel = copy.deepcopy(program.kernels[op.kernel])
            kernel.macros.update(GQA_CHUNK_SG=f'{partition_sg}u', STATIC_GQA_P_N_SG='192u')
            program.kernels[key] = kernel
        op.kernel = key
        data = bytearray(program.buffers[name].init)
        struct.pack_into('<I', data, offset + 16, 192)
        program.buffers[name].init = bytes(data)
        if program.kernels[key].function == 'gqa_decode':
            op.grid, op.threadgroup = (24, 1, 1), (256, 1, 1)
    used = {op.kernel for op in program.ops}
    for key in replaced - used:
        del program.kernels[key]
    return program
