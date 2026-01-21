#!/usr/bin/env python3
"""
Benchmark runner to compare all kernel versions.

Usage:
    python benchmark_all.py           # Run all versions once
    python benchmark_all.py --runs 5  # Run all versions 5 times each
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
    N_CORES,
)


@dataclass
class BenchmarkResult:
    name: str
    cycles: list[int]
    instruction_count: int
    wall_times: list[float]
    correct: bool

    @property
    def min_cycles(self) -> int:
        return min(self.cycles)

    @property
    def max_cycles(self) -> int:
        return max(self.cycles)

    @property
    def mean_cycles(self) -> float:
        return sum(self.cycles) / len(self.cycles)

    @property
    def speedup(self) -> float:
        return 147734 / self.mean_cycles


def run_kernel_test(kb_module, seed: int = 123) -> tuple[int, int, bool]:
    """Run a kernel and return (cycles, instruction_count, correct)"""
    random.seed(seed)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    kb = kb_module.KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
    )
    machine.enable_pause = False
    machine.enable_debug = False

    # Run and get reference
    machine.run()

    # Verify correctness
    ref_mem = list(reference_kernel2(mem, {}))[-1]
    inp_values_p = ref_mem[6]
    correct = (
        machine.mem[inp_values_p : inp_values_p + 256]
        == ref_mem[inp_values_p : inp_values_p + 256]
    )

    return machine.cycle, len(kb.instrs), correct


def benchmark_version(name: str, module, runs: int, seeds: list[int]) -> BenchmarkResult:
    """Benchmark a kernel version with multiple runs"""
    cycles = []
    wall_times = []
    instruction_count = 0
    all_correct = True

    for i, seed in enumerate(seeds[:runs]):
        start = time.time()
        cyc, instr_count, correct = run_kernel_test(module, seed)
        elapsed = time.time() - start

        cycles.append(cyc)
        wall_times.append(elapsed)
        instruction_count = instr_count
        if not correct:
            all_correct = False

    return BenchmarkResult(
        name=name,
        cycles=cycles,
        instruction_count=instruction_count,
        wall_times=wall_times,
        correct=all_correct,
    )


def print_results(results: list[BenchmarkResult]):
    """Pretty print benchmark results"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Version':<25} {'Status':<8} {'Instrs':>8} {'Min':>10} {'Mean':>10} {'Max':>10} {'Speedup':>8}")
    print("-" * 80)

    baseline_cycles = 147734
    for r in results:
        status = "PASS" if r.correct else "FAIL"
        print(
            f"{r.name:<25} {status:<8} {r.instruction_count:>8} "
            f"{r.min_cycles:>10} {r.mean_cycles:>10.1f} {r.max_cycles:>10} "
            f"{r.speedup:>7.2f}x"
        )

    print("-" * 80)
    print(f"{'Baseline (reference)':<25} {'N/A':<8} {'N/A':>8} {baseline_cycles:>10} {baseline_cycles:>10.1f} {baseline_cycles:>10} {'1.00':>7}x")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark all kernel versions")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per version")
    args = parser.parse_args()

    # Generate seeds for reproducible random tests
    seeds = [123, 456, 789, 101112, 42, 999, 314159, 271828, 141421, 173205]

    print(f"Running benchmarks with {args.runs} run(s) each...")
    print(f"Test parameters: forest_height=10, rounds=16, batch_size=256")

    results = []

    # Import and test each version
    versions = [
        ("v0_baseline", "perf_takehome"),
        ("v1_loops", "perf_takehome_v1_loops"),
        ("v2_vectorized", "perf_takehome_v2_vectorized"),
        ("v3_vliw", "perf_takehome_v3_vliw"),
        ("v4_pipelined", "perf_takehome_v4_pipelined"),
    ]

    for name, module_name in versions:
        try:
            module = __import__(module_name)
            print(f"\nBenchmarking {name}...")
            result = benchmark_version(name, module, args.runs, seeds)
            results.append(result)
        except ImportError as e:
            print(f"  Skipping {name}: {e}")
        except Exception as e:
            print(f"  Error in {name}: {e}")

    print_results(results)

    # Summary
    print("\nSummary:")
    for r in results:
        if r.correct:
            print(f"  {r.name}: {r.mean_cycles:.0f} cycles ({r.speedup:.2f}x vs baseline)")
        else:
            print(f"  {r.name}: INCORRECT OUTPUT")


if __name__ == "__main__":
    main()
