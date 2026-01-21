"""
Version 3: VLIW Packing + Vectorization + Constant Hoisting

Changes from v2:
- Pack multiple independent operations into same cycle (VLIW)
- Hash: pack tmp1 and tmp2 computation (6 stages × 2 cycles = 12, was 18)
- Gather: pack multiple address calculations
- Better instruction scheduling overall

Expected improvement: ~1.5x over v2 from reduced cycle count
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
        """Add a single-slot instruction (legacy compatibility)"""
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle: dict):
        """Add a VLIW instruction bundle with multiple slots per engine"""
        # Validate slot limits
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
        """Allocate a VLEN-sized vector in scratch"""
        return self.alloc_scratch(name, VLEN)

    def scratch_const(self, val, name=None):
        """Load a scalar constant into scratch (cached)"""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vec_const(self, val, name=None):
        """Load a vector constant (broadcast scalar to vector, cached)"""
        if val not in self.vec_const_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_vec(name)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.vec_const_map[val] = vec_addr
        return self.vec_const_map[val]

    def build_hash_vec_packed(self, val_vec, tmp1_vec, tmp2_vec):
        """Build vectorized hash with VLIW packing (2 cycles per stage instead of 3)"""
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            const1_vec = self.vec_const(val1)
            const3_vec = self.vec_const(val3)
            # Pack: tmp1 = val OP1 const1 AND tmp2 = val OP3 const3 (independent!)
            self.add_bundle({
                "valu": [
                    (op1, tmp1_vec, val_vec, const1_vec),
                    (op3, tmp2_vec, val_vec, const3_vec),
                ]
            })
            # val = tmp1 OP2 tmp2 (depends on both)
            self.add("valu", (op2, val_vec, tmp1_vec, tmp2_vec))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        VLIW-packed vectorized kernel.
        """
        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")

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

        # Scalar constants
        zero_const = self.scratch_const(0, "zero")
        one_const = self.scratch_const(1, "one")
        two_const = self.scratch_const(2, "two")
        vlen_const = self.scratch_const(VLEN, "vlen")

        # Preload ALL hash constants as vectors BEFORE the loop
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.vec_const(val1)
            self.vec_const(val3)

        # Vector constants
        zero_vec = self.vec_const(0, "zero_vec")
        one_vec = self.vec_const(1, "one_vec")
        two_vec = self.vec_const(2, "two_vec")

        # Broadcast n_nodes to vector
        n_nodes_vec = self.alloc_vec("n_nodes_vec")
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        # Pause for debug sync
        self.add("flow", ("pause",))

        # Loop counters
        round_counter = self.alloc_scratch("round_counter")
        chunk_counter = self.alloc_scratch("chunk_counter")
        chunk_addr = self.alloc_scratch("chunk_addr")

        # Number of chunks
        n_chunks = self.alloc_scratch("n_chunks")
        self.add("alu", ("//", n_chunks, self.scratch["batch_size"], vlen_const))

        # Working vector registers
        idx_vec = self.alloc_vec("idx_vec")
        val_vec = self.alloc_vec("val_vec")
        node_val_vec = self.alloc_vec("node_val_vec")
        tmp1_vec = self.alloc_vec("tmp1_vec")
        tmp2_vec = self.alloc_vec("tmp2_vec")
        cond_vec = self.alloc_vec("cond_vec")

        # Multiple address temporaries for gather packing
        addr_temps = [self.alloc_scratch(f"addr_tmp{i}") for i in range(VLEN)]
        tmp_addr = self.alloc_scratch("tmp_addr")
        cond = self.alloc_scratch("cond")

        # Initialize outer loop
        self.add("load", ("const", round_counter, 0))

        # === OUTER LOOP START ===
        outer_loop_start = len(self.instrs)

        # Initialize inner loop
        self.add_bundle({
            "load": [
                ("const", chunk_counter, 0),
                ("const", chunk_addr, 0),
            ]
        })

        # === INNER LOOP START ===
        inner_loop_start = len(self.instrs)

        # --- Load indices and values (pack address calc with previous vload if possible) ---
        # Compute both addresses, then do both vloads
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

        # --- Gather: compute all 8 addresses, then load in pairs ---
        # Compute addresses for elements 0-3 (pack 4 ALU ops)
        self.add_bundle({
            "alu": [
                ("+", addr_temps[0], self.scratch["forest_values_p"], idx_vec + 0),
                ("+", addr_temps[1], self.scratch["forest_values_p"], idx_vec + 1),
                ("+", addr_temps[2], self.scratch["forest_values_p"], idx_vec + 2),
                ("+", addr_temps[3], self.scratch["forest_values_p"], idx_vec + 3),
            ]
        })
        # Compute addresses for elements 4-7
        self.add_bundle({
            "alu": [
                ("+", addr_temps[4], self.scratch["forest_values_p"], idx_vec + 4),
                ("+", addr_temps[5], self.scratch["forest_values_p"], idx_vec + 5),
                ("+", addr_temps[6], self.scratch["forest_values_p"], idx_vec + 6),
                ("+", addr_temps[7], self.scratch["forest_values_p"], idx_vec + 7),
            ]
        })
        # Load pairs (2 loads/cycle limit)
        self.add_bundle({"load": [("load", node_val_vec + 0, addr_temps[0]), ("load", node_val_vec + 1, addr_temps[1])]})
        self.add_bundle({"load": [("load", node_val_vec + 2, addr_temps[2]), ("load", node_val_vec + 3, addr_temps[3])]})
        self.add_bundle({"load": [("load", node_val_vec + 4, addr_temps[4]), ("load", node_val_vec + 5, addr_temps[5])]})
        self.add_bundle({"load": [("load", node_val_vec + 6, addr_temps[6]), ("load", node_val_vec + 7, addr_temps[7])]})

        # --- XOR ---
        self.add("valu", ("^", val_vec, val_vec, node_val_vec))

        # --- Hash (packed: 2 cycles per stage) ---
        self.build_hash_vec_packed(val_vec, tmp1_vec, tmp2_vec)

        # --- Compute new indices (pack where possible) ---
        # tmp1_vec = val_vec % 2; cond_vec = (tmp1_vec == 0)
        self.add_bundle({
            "valu": [
                ("%", tmp1_vec, val_vec, two_vec),
            ]
        })
        self.add_bundle({
            "valu": [
                ("==", cond_vec, tmp1_vec, zero_vec),
                ("*", idx_vec, idx_vec, two_vec),  # Can compute in parallel
            ]
        })
        # tmp2_vec = cond_vec ? 1 : 2
        self.add("flow", ("vselect", tmp2_vec, cond_vec, one_vec, two_vec))
        # idx_vec = idx_vec + tmp2_vec
        self.add("valu", ("+", idx_vec, idx_vec, tmp2_vec))

        # Wrap around: idx_vec = (idx_vec < n_nodes) ? idx_vec : 0
        self.add("valu", ("<", cond_vec, idx_vec, n_nodes_vec))
        self.add("flow", ("vselect", idx_vec, cond_vec, idx_vec, zero_vec))

        # --- Store (pack address calc with stores) ---
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

        # --- Inner loop footer (pack where possible) ---
        self.add_bundle({
            "flow": [
                ("add_imm", chunk_counter, chunk_counter, 1),
            ],
            "alu": [
                ("+", chunk_addr, chunk_addr, vlen_const),  # Use ALU instead of add_imm
            ]
        })
        self.add("alu", ("<", cond, chunk_counter, n_chunks))
        self.add("flow", ("cond_jump", cond, inner_loop_start))

        # === INNER LOOP END ===

        # --- Outer loop footer ---
        self.add("flow", ("add_imm", round_counter, round_counter, 1))
        self.add("alu", ("<", cond, round_counter, self.scratch["rounds"]))
        self.add("flow", ("cond_jump", cond, outer_loop_start))

        # === OUTER LOOP END ===
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
        for seed in [123, 456, 789, 101112]:
            do_kernel_test(10, 16, 256, seed=seed)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True)


if __name__ == "__main__":
    unittest.main()
