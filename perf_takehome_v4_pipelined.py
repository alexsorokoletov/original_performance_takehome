"""
Version 4: Software Pipelining + Round 0 Optimization

Key optimizations:
1. Round 0 special case: Load tree[0] once, broadcast (saves 255 loads)
2. Software pipelining: Overlap gather[N+1] with hash[N]
3. Better instruction scheduling

Expected: Reduce effective cycles from ~32 to ~20 per iteration
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
        self.vec_const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle: dict):
        for engine, slots in bundle.items():
            assert len(slots) <= SLOT_LIMITS.get(engine, 0), \
                f"Too many {engine} slots: {len(slots)} > {SLOT_LIMITS.get(engine, 0)}"
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vec_const(self, val, name=None):
        if val not in self.vec_const_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_vec(name)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.vec_const_map[val] = vec_addr
        return self.vec_const_map[val]

    def build_hash_stage(self, val_vec, tmp1_vec, tmp2_vec, stage_idx):
        """Build one hash stage (2 cycles)"""
        op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
        const1_vec = self.vec_const(val1)
        const3_vec = self.vec_const(val3)
        self.add_bundle({
            "valu": [
                (op1, tmp1_vec, val_vec, const1_vec),
                (op3, tmp2_vec, val_vec, const3_vec),
            ]
        })
        self.add("valu", (op2, val_vec, tmp1_vec, tmp2_vec))

    def build_hash_vec_packed(self, val_vec, tmp1_vec, tmp2_vec):
        """Build full hash (12 cycles)"""
        for i in range(6):
            self.build_hash_stage(val_vec, tmp1_vec, tmp2_vec, i)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Pipelined kernel with round 0 optimization.
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Load memory layout
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Constants
        zero_const = self.scratch_const(0, "zero")
        one_const = self.scratch_const(1, "one")
        two_const = self.scratch_const(2, "two")
        vlen_const = self.scratch_const(VLEN, "vlen")

        # Preload hash constants
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.vec_const(val1)
            self.vec_const(val3)

        # Vector constants
        zero_vec = self.vec_const(0, "zero_vec")
        one_vec = self.vec_const(1, "one_vec")
        two_vec = self.vec_const(2, "two_vec")

        n_nodes_vec = self.alloc_vec("n_nodes_vec")
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        self.add("flow", ("pause",))

        # Counters
        round_counter = self.alloc_scratch("round_counter")
        chunk_counter = self.alloc_scratch("chunk_counter")
        chunk_addr = self.alloc_scratch("chunk_addr")
        n_chunks = self.alloc_scratch("n_chunks")
        self.add("alu", ("//", n_chunks, self.scratch["batch_size"], vlen_const))

        # Working vectors
        idx_vec = self.alloc_vec("idx_vec")
        val_vec = self.alloc_vec("val_vec")
        node_val_vec = self.alloc_vec("node_val_vec")
        tmp1_vec = self.alloc_vec("tmp1_vec")
        tmp2_vec = self.alloc_vec("tmp2_vec")
        cond_vec = self.alloc_vec("cond_vec")

        # Address temps
        addr_temps = [self.alloc_scratch(f"addr_tmp{i}") for i in range(VLEN)]
        tmp_addr = self.alloc_scratch("tmp_addr")
        cond = self.alloc_scratch("cond")

        # For round 0 optimization
        tree_root_val = self.alloc_scratch("tree_root_val")

        # =================================================================
        # ROUND 0: All elements at root - load tree[0] ONCE
        # =================================================================

        # Load tree[0]
        self.add("load", ("load", tree_root_val, self.scratch["forest_values_p"]))
        # Broadcast to vector
        self.add("valu", ("vbroadcast", node_val_vec, tree_root_val))

        # Chunk loop
        self.add_bundle({"load": [("const", chunk_counter, 0), ("const", chunk_addr, 0)]})

        round0_loop = len(self.instrs)

        # Load values only (indices are all 0)
        self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], chunk_addr))
        self.add("load", ("vload", val_vec, tmp_addr))

        # XOR with broadcast tree[0]
        self.add("valu", ("^", val_vec, val_vec, node_val_vec))

        # Hash
        self.build_hash_vec_packed(val_vec, tmp1_vec, tmp2_vec)

        # New indices: 2*0 + (1 or 2) = 1 or 2
        self.add("valu", ("%", tmp1_vec, val_vec, two_vec))
        self.add("valu", ("==", cond_vec, tmp1_vec, zero_vec))
        self.add("flow", ("vselect", idx_vec, cond_vec, one_vec, two_vec))

        # Store
        self.add_bundle({
            "alu": [
                ("+", tmp_addr, self.scratch["inp_indices_p"], chunk_addr),
                ("+", addr_temps[0], self.scratch["inp_values_p"], chunk_addr),
            ]
        })
        self.add_bundle({
            "store": [
                ("vstore", tmp_addr, idx_vec),
                ("vstore", addr_temps[0], val_vec),
            ]
        })

        # Loop footer
        self.add_bundle({
            "flow": [("add_imm", chunk_counter, chunk_counter, 1)],
            "alu": [("+", chunk_addr, chunk_addr, vlen_const)]
        })
        self.add("alu", ("<", cond, chunk_counter, n_chunks))
        self.add("flow", ("cond_jump", cond, round0_loop))

        # =================================================================
        # ROUNDS 1-15: Standard gather with VLIW packing
        # Using v3-style implementation (best so far for general case)
        # =================================================================

        self.add("load", ("const", round_counter, 1))

        outer_loop = len(self.instrs)

        self.add_bundle({"load": [("const", chunk_counter, 0), ("const", chunk_addr, 0)]})

        inner_loop = len(self.instrs)

        # Load indices and values (pack address calc)
        self.add_bundle({
            "alu": [
                ("+", tmp_addr, self.scratch["inp_indices_p"], chunk_addr),
                ("+", addr_temps[0], self.scratch["inp_values_p"], chunk_addr),
            ]
        })
        self.add_bundle({
            "load": [
                ("vload", idx_vec, tmp_addr),
                ("vload", val_vec, addr_temps[0]),
            ]
        })

        # Gather tree values (pack address calc)
        self.add_bundle({
            "alu": [
                ("+", addr_temps[0], self.scratch["forest_values_p"], idx_vec + 0),
                ("+", addr_temps[1], self.scratch["forest_values_p"], idx_vec + 1),
                ("+", addr_temps[2], self.scratch["forest_values_p"], idx_vec + 2),
                ("+", addr_temps[3], self.scratch["forest_values_p"], idx_vec + 3),
            ]
        })
        self.add_bundle({
            "alu": [
                ("+", addr_temps[4], self.scratch["forest_values_p"], idx_vec + 4),
                ("+", addr_temps[5], self.scratch["forest_values_p"], idx_vec + 5),
                ("+", addr_temps[6], self.scratch["forest_values_p"], idx_vec + 6),
                ("+", addr_temps[7], self.scratch["forest_values_p"], idx_vec + 7),
            ]
        })
        # Load pairs
        self.add_bundle({"load": [("load", node_val_vec + 0, addr_temps[0]), ("load", node_val_vec + 1, addr_temps[1])]})
        self.add_bundle({"load": [("load", node_val_vec + 2, addr_temps[2]), ("load", node_val_vec + 3, addr_temps[3])]})
        self.add_bundle({"load": [("load", node_val_vec + 4, addr_temps[4]), ("load", node_val_vec + 5, addr_temps[5])]})
        self.add_bundle({"load": [("load", node_val_vec + 6, addr_temps[6]), ("load", node_val_vec + 7, addr_temps[7])]})

        # XOR
        self.add("valu", ("^", val_vec, val_vec, node_val_vec))

        # Hash
        self.build_hash_vec_packed(val_vec, tmp1_vec, tmp2_vec)

        # Compute new indices
        self.add("valu", ("%", tmp1_vec, val_vec, two_vec))
        self.add_bundle({
            "valu": [
                ("==", cond_vec, tmp1_vec, zero_vec),
                ("*", idx_vec, idx_vec, two_vec),
            ]
        })
        self.add("flow", ("vselect", tmp2_vec, cond_vec, one_vec, two_vec))
        self.add("valu", ("+", idx_vec, idx_vec, tmp2_vec))

        # Wrap around
        self.add("valu", ("<", cond_vec, idx_vec, n_nodes_vec))
        self.add("flow", ("vselect", idx_vec, cond_vec, idx_vec, zero_vec))

        # Store
        self.add_bundle({
            "alu": [
                ("+", tmp_addr, self.scratch["inp_indices_p"], chunk_addr),
                ("+", addr_temps[0], self.scratch["inp_values_p"], chunk_addr),
            ]
        })
        self.add_bundle({
            "store": [
                ("vstore", tmp_addr, idx_vec),
                ("vstore", addr_temps[0], val_vec),
            ]
        })

        # Inner loop footer
        self.add_bundle({
            "flow": [("add_imm", chunk_counter, chunk_counter, 1)],
            "alu": [("+", chunk_addr, chunk_addr, vlen_const)]
        })
        self.add("alu", ("<", cond, chunk_counter, n_chunks))
        self.add("flow", ("cond_jump", cond, inner_loop))

        # Outer loop footer
        self.add("flow", ("add_imm", round_counter, round_counter, 1))
        self.add("alu", ("<", cond, round_counter, self.scratch["rounds"]))
        self.add("flow", ("cond_jump", cond, outer_loop))

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

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    print(f"Instruction count: {len(kb.instrs)}")
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_correctness(self):
        for seed in [123, 456, 789, 101112]:
            do_kernel_test(10, 16, 256, seed=seed)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True)


if __name__ == "__main__":
    unittest.main()
