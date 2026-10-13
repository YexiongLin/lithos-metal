# Apple M6 (12 GPU cores)

Measured on Apple11, 32 GB RAM, macOS 27.0.1 (2026-10-13). Selection requires
the exact chip name, GPU family and core count; other M6 variants are not
qualified by this profile. All Metal templates come from `kernels/common/`.

The profile enables tensor acceleration with the existing interleaved16 weight
layout. BF16 matrices retain the shader path below 16 rows, and attention stays
on v1: broader acceleration and alternate attention paths failed strict
sequence-equivalence checks. `config_kernels.json` contains the qualified
per-shape launch choices. No M5 fusion recipe or speculative cost table is used.

Matching dynamic T=8/32/128 attention programs use 24 threadgroups of eight
SIMD groups. The `GQA_CHUNK_SG` macro preserves the original 144-SIMD softmax
partition in both core and merge, independently of the new launch. The policy
is limited to target heads/KV heads/dimension 24/4/256 and seven-proposal draft
32/8/128; other shapes and static programs keep existing launches.

Serving uses 128-token prefill and caps speculative context at 8192. On this
machine, 16K generation works but host prefix caching can induce swapping;
32K with 128-row prefill exceeds GPU residency. KV caches remain BF16.

## Validation and performance [M]

The workload was `nvidia/Qwen3.8-27B-NVFP4` with the NVFP4 DSpark draft.
Fresh-process A/B/B/A runs compare the prior 12-by-12 attention launch with
24-by-8, holding weights, projections, softmax partitions and output tokens
constant. Each request uses an 8,064-token prompt and generates 128 tokens,
with prefix caching disabled. Timings exclude compilation and shader validation.

| Run | Prefill GPU time | Decode throughput | Total wall time |
| --- | ---: | ---: | ---: |
| A1 | 65.308 s | 13.175 tok/s | 74.977 s |
| B1 | 63.717 s | 13.604 tok/s | 73.080 s |
| B2 | 63.771 s | 13.581 tok/s | 73.152 s |
| A2 | 65.323 s | 13.122 tok/s | 75.028 s |

There is no short-context speedup claim. At position 128, alternating fixed-state
rounds took 184.39/185.02, 185.04/185.75 and 185.83/186.32 ms (A/B), a small
short-context cost. The original 1,024-token nonstreaming HTTP request completed
in 57.05 seconds (17.95 tok/s), with the same message as the previous version.

Regression coverage checks device selection, serving limits, recipe identity,
core/merge parameters, exact target/draft outputs and cache writes, and the
small-K NVFP4 Markov-head schedule. GPU checks use Metal shader validation.
Experimental benchmark scripts and detailed research logs are not required by
the backend and are excluded from this change.
