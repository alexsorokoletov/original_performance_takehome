"""
Version 1: Loop Optimization

Changes from baseline:
- Replace fully unrolled loops with actual loop instructions using cond_jump
- Reduces code size from ~147k instructions to ~100 instructions
- Uses flow engine for loop control

Expected improvement: Code size reduction, slight cycle improvement from reduced instruction fetch
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2):
        """Build hash computation without debug statements for looped version"""
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Loop-optimized kernel using actual jump instructions instead of unrolling.
        """
        # Allocate temporary registers
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Load memory layout from header
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Common constants
        zero_const = self.scratch_const(0, "zero")
        one_const = self.scratch_const(1, "one")
        two_const = self.scratch_const(2, "two")

        # IMPORTANT: Preload ALL hash constants BEFORE the loop starts
        # Otherwise they get placed inside the loop and executed every iteration
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.scratch_const(val1)
            self.scratch_const(val3)

        # Pause for debug sync (ignored in submission)
        self.add("flow", ("pause",))

        # Allocate loop counters
        round_counter = self.alloc_scratch("round_counter")
        batch_counter = self.alloc_scratch("batch_counter")

        # Allocate working registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        cond = self.alloc_scratch("cond")

        # Initialize outer loop counter: round_counter = 0
        self.add("load", ("const", round_counter, 0))

        # === OUTER LOOP START (rounds) ===
        outer_loop_start = len(self.instrs)

        # Initialize inner loop counter: batch_counter = 0
        self.add("load", ("const", batch_counter, 0))

        # === INNER LOOP START (batch) ===
        inner_loop_start = len(self.instrs)

        # --- Loop body: process one element ---

        # idx = mem[inp_indices_p + batch_counter]
        self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], batch_counter))
        self.add("load", ("load", tmp_idx, tmp_addr))

        # val = mem[inp_values_p + batch_counter]
        self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], batch_counter))
        self.add("load", ("load", tmp_val, tmp_addr))

        # node_val = mem[forest_values_p + idx]
        self.add("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
        self.add("load", ("load", tmp_node_val, tmp_addr))

        # val = val ^ node_val
        self.add("alu", ("^", tmp_val, tmp_val, tmp_node_val))

        # val = myhash(val)
        hash_slots = self.build_hash(tmp_val, tmp1, tmp2)
        for engine, slot in hash_slots:
            self.add(engine, slot)

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        self.add("alu", ("%", tmp1, tmp_val, two_const))
        self.add("alu", ("==", tmp1, tmp1, zero_const))
        self.add("flow", ("select", tmp3, tmp1, one_const, two_const))
        self.add("alu", ("*", tmp_idx, tmp_idx, two_const))
        self.add("alu", ("+", tmp_idx, tmp_idx, tmp3))

        # idx = 0 if idx >= n_nodes else idx
        self.add("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]))
        self.add("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const))

        # mem[inp_indices_p + batch_counter] = idx
        self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], batch_counter))
        self.add("store", ("store", tmp_addr, tmp_idx))

        # mem[inp_values_p + batch_counter] = val
        self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], batch_counter))
        self.add("store", ("store", tmp_addr, tmp_val))

        # --- End loop body ---

        # Inner loop: batch_counter++; if batch_counter < batch_size goto inner_loop_start
        self.add("flow", ("add_imm", batch_counter, batch_counter, 1))
        self.add("alu", ("<", cond, batch_counter, self.scratch["batch_size"]))
        self.add("flow", ("cond_jump", cond, inner_loop_start))

        # === INNER LOOP END ===

        # Outer loop: round_counter++; if round_counter < rounds goto outer_loop_start
        self.add("flow", ("add_imm", round_counter, round_counter, 1))
        self.add("alu", ("<", cond, round_counter, self.scratch["rounds"]))
        self.add("flow", ("cond_jump", cond, outer_loop_start))

        # === OUTER LOOP END ===

        # Final pause for debug sync
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    print(f"Instruction count: {len(kb.instrs)}")
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_correctness(self):
        """Test with multiple random seeds"""
        for seed in [123, 456, 789, 101112]:
            do_kernel_test(10, 16, 256, seed=seed)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True)


if __name__ == "__main__":
    unittest.main()
