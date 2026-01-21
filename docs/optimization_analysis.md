# Performance Optimization Analysis

## Problem Overview

**Task**: Optimize a binary tree traversal kernel on a simulated VLIW SIMD architecture.

**Algorithm**:
```python
for round in range(16):
    for i in range(256):
        idx = indices[i]
        val = values[i]
        val = myhash(val ^ tree[idx])
        idx = 2*idx + (1 if val%2==0 else 2)
        idx = 0 if idx >= n_nodes else idx
        indices[i] = idx
        values[i] = val
```

**Hash Function**: 6 stages, each: `a = (a OP1 const1) OP2 (a OP3 const3)`

---

## Architecture Constraints

| Resource | Limit per Cycle | Notes |
|----------|-----------------|-------|
| ALU slots | 12 | Arithmetic/logic operations |
| VALU slots | 6 | Vector operations (VLEN=8) |
| Load slots | 2 | Memory read |
| Store slots | 2 | Memory write |
| Flow slots | 1 | Jumps, selects |
| Scratch | 1536 words | Register-like storage |
| VLEN | 8 | SIMD width |

---

## Classic ASM Championship Analogies

| Pattern | Our Problem | Key Insight |
|---------|-------------|-------------|
| **Linked List Traversal** | Tree index depends on hash | Serial dependency chain |
| **Sparse Matrix-Vector (SpMV)** | Scatter-gather pattern | Indirect memory bottleneck |
| **Histogram/Reduction** | Many elements → few tree nodes | DUPLICATION opportunity |
| **Radix Sort** | Level-by-level processing | Level-aware batching |
| **Binary Indexed Tree (BIT)** | Shared prefix paths | Path coalescing |
| **Hash Table Probing** | Hash → lookup → repeat | Dependent address generation |
| **Graph BFS** | Follow edges based on values | Pointer chasing |

---

## Tree Structure Observations

### Standard Binary Tree Properties
- Height: 10 (n_nodes = 2047)
- Heap layout: children at 2*idx+1, 2*idx+2
- Static (never changes during execution)

### Level Convergence Pattern
```
Round 0:  ALL 256 elements at idx=0 (root) → 1 unique node
Round 1:  Elements split to idx=1,2        → 2 unique nodes
Round 2:  Elements at idx=3,4,5,6          → 4 unique nodes
Round 3:  Elements at idx=7..14            → 8 unique nodes
...
Round 7:  Elements at idx=127..254         → 128 unique nodes
Round 8+: Elements scattered               → 256 unique nodes (saturated)
```

**Key insight**: In round 0, we load tree[0] for ALL 256 elements.
That's 255 redundant loads!

### Alternative Tree Structures Considered

| Structure | Pros | Cons | Applicable? |
|-----------|------|------|-------------|
| **B-tree** | Cache-friendly, multiple keys/node | Complex traversal | No - algorithm fixed |
| **Van Emde Boas layout** | Cache-oblivious locality | Reordering overhead | Maybe - for caching |
| **Radix/Trie** | Path compression | Different structure | No - algorithm fixed |
| **Cache-blocked** | Fits in scratch | Extra indirection | Yes - partial caching |

---

## Architectural Capacity Analysis

### Current Resource Utilization (v3)

| Resource | Limit | Current Usage | Utilization | Status |
|----------|-------|---------------|-------------|--------|
| **Load** | 2/cycle | 2/cycle | 100% | **BOTTLENECK** |
| **Store** | 2/cycle | ~0.5/cycle | 25% | Free |
| **ALU** | 12/cycle | ~2/cycle | 17% | Very free |
| **VALU** | 6/cycle | ~2/cycle | 33% | Free |
| **Flow** | 1/cycle | ~0.2/cycle | 20% | Free |
| **Scratch** | 1536 words | ~250 words | 16% | **1300 FREE** |

### Bottleneck Identification

**Primary**: Memory bandwidth (2 loads/cycle)
**Secondary**: Hash dependency chain (12 cycles minimum)
**Not a bottleneck**: Compute (ALU/VALU), Scratch space

### Resource Opportunity Matrix

| Have Free | Can Use For |
|-----------|-------------|
| Scratch (1300 words) | Cache tree levels, pipeline buffers, more vectors |
| ALU (10 slots) | Parallel address computation, loop overhead |
| VALU (4 slots) | Multi-chunk hash interleaving |
| Store (1.5 slots) | Negligible benefit |

---

## Load Analysis

### Current Approach (v3)
- Total tree loads: 256 elements × 16 rounds = 4096
- Load cycles: 4096 / 2 per cycle = 2048 cycles minimum

### With Level-Aware Deduplication
```
Rounds 0-7 (can deduplicate):
  Level 0: 1 load
  Level 1: 2 loads
  Level 2: 4 loads
  Level 3: 8 loads
  Level 4: 16 loads
  Level 5: 32 loads
  Level 6: 64 loads
  Level 7: 128 loads
  Subtotal: 255 loads

Rounds 8-15 (256 unique each, worst case):
  8 rounds × 256 = 2048 loads

Total: 2303 loads → 1152 cycles minimum
```

**Reduction**: 4096 → 2303 loads (44% reduction!)

### Load Categories

| Category | Count | Cycles | Can Optimize? |
|----------|-------|--------|---------------|
| Tree (rounds 0-7) | 255 | 128 | YES - level loading |
| Tree (rounds 8-15) | 2048 | 1024 | Partial - some sharing |
| Batch indices (vload) | 512 | 256 | No - required |
| Batch values (vload) | 512 | 256 | No - required |
| **Total** | **3327** | **1664** | |

---

## Hash Pipeline Analysis

### Current Structure (v3)
```
Per stage (6 total):
  Cycle 1: tmp1 = val OP1 const1 | tmp2 = val OP3 const3 (2 VALU)
  Cycle 2: val = tmp1 OP2 tmp2                           (1 VALU)

Total: 6 stages × 2 cycles = 12 cycles per chunk (8 elements)
```

### With 3-Way VALU Interleaving
```
Process 3 chunks (24 elements) simultaneously:
  Cycle 1: [A.tmp1, A.tmp2, B.tmp1, B.tmp2, C.tmp1, C.tmp2] (6 VALU - maxed!)
  Cycle 2: [A.comb, B.comb, C.comb, A'.tmp1, A'.tmp2, B'.tmp1]
  ...

Result: 12 cycles per 3 chunks = 4 cycles per chunk
```

**Benefit**: 3× hash throughput (if not load-bound)

### Hash Cannot Be Reduced Because
1. 6 stages are algorithmic requirement
2. Each stage has serial dependency (combine depends on tmp1, tmp2)
3. We already pack the independent operations

---

## Software Pipelining Strategy

### Pipeline Stages
```
Stage A: Gather tree values (4 cycles for 8 loads)
Stage B: Hash computation (12 cycles)
Stage C: Index computation + store (4 cycles)
```

### Overlapped Execution
```
Cycle 1-4:   Gather[N]   | Hash[N-1]    | Store[N-2]
Cycle 5-8:   Gather[N]   | Hash[N-1]    | Store[N-2]
Cycle 9-12:  Gather[N+1] | Hash[N]      | Store[N-1]
...
```

### Pipeline Depth Requirements
- Need scratch for 3 iterations in flight
- Per iteration: ~6 vectors × 8 words = 48 words
- Total: 144 words (fits easily in 1300 free)

---

## Optimization Strategies

### Implemented (v1-v3)

| Version | Strategy | Cycles | Speedup |
|---------|----------|--------|---------|
| v0 | Baseline (unrolled) | 147,734 | 1.00x |
| v1 | Loop structure | 159,840 | 0.92x |
| v2 | SIMD vectorization | 27,778 | 5.32x |
| v3 | VLIW packing | 16,498 | 8.95x |

### Proposed (v4+)

| Version | Strategy | Expected Cycles | Expected Speedup |
|---------|----------|-----------------|------------------|
| v4 | Level-aware + pipelining | ~4,000-6,000 | ~25-35x |
| v5 | Loop unrolling + VALU interleave | ~2,000-3,000 | ~50-70x |
| v6 | Micro-optimization | ~1,500-2,000 | ~75-100x |

---

## Theoretical Minimum Analysis

### Absolute Lower Bounds
```
Unique tree loads:     2303 / 2 = 1152 cycles
Batch I/O:             1024 / 2 = 512 cycles (vload+vstore)
Hash (if overlapped):  0 additional (hidden by loads)
Loop overhead:         ~50 cycles (heavily unrolled)
────────────────────────────────────────────────
Theoretical minimum:   ~1714 cycles
```

### Why <1500 Might Be Achievable
1. Some tree sharing in rounds 8-15 (pigeonhole)
2. Better overlap than calculated
3. Algorithmic tricks we haven't discovered

### Why <1500 Might Be Hard
1. 2 loads/cycle is fundamental limit
2. Hash serial dependency can't be eliminated
3. Some overhead is unavoidable

---

## Implementation Roadmap

### v4: Level-Aware Loading + Software Pipelining
```
PROLOGUE:
  Load tree levels 0-7 into scratch (255 loads)

ROUNDS 0-7 (level-aware):
  For each level:
    Load all 2^level tree nodes
    For each chunk:
      Select correct tree value per element
      Hash (pipelined with next level's load)
      Store results

ROUNDS 8-15 (gather with pipelining):
  Deep software pipeline
  Gather[N] || Hash[N-1] || Store[N-2]
```

### v5: Multi-Chunk Processing
- Process 2-4 chunks per iteration
- Fully utilize VALU slots (6)
- Amortize loop overhead

### v6: Final Optimizations
- Unroll loops 4-8×
- Perfect instruction scheduling
- Minimize all overhead

---

## Alternative Approaches Not Explored

### Speculative Prefetching
Load BOTH children before knowing direction:
- Pro: Removes dependency on hash result
- Con: 2× tree loads (worse for bandwidth-bound)

### Path Memoization
Precompute common traversal paths:
- Pro: Could share computation
- Con: Paths depend on runtime values

### Tree Reordering
Rearrange tree for better locality:
- Pro: Better cache behavior
- Con: Need scatter/gather which we don't have

### Batch Sorting
Sort batch by current index to coalesce loads:
- Pro: Better memory access pattern
- Con: Sort overhead, destroys vectorization

---

## Key Takeaways

1. **We're memory-bound** - Load throughput (2/cycle) is the bottleneck
2. **77% redundant loads** in rounds 0-7 due to level convergence
3. **1300 words scratch free** - Should use for caching/pipelining
4. **VALU underutilized** - Can interleave multiple chunks
5. **Level-aware loading** is the key optimization
6. **Software pipelining** can hide remaining latency

---

## Benchmark Targets

| Target | Cycles | Speedup | Status |
|--------|--------|---------|--------|
| Baseline | 147,734 | 1.0x | Reference |
| Updated 2-hr start | 18,532 | 8.0x | ✅ PASSED (v3) |
| Opus 4 many hours | 2,164 | 68x | In progress |
| Opus 4.5 casual | 1,790 | 83x | Target |
| Opus 4.5 2-hours | 1,579 | 94x | Stretch |
| Opus 4.5 11.5-hours | 1,487 | 99x | Aspirational |
| Best known | 1,363 | 108x | Ultimate |
