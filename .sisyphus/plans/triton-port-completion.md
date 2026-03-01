# aihwkit Triton Backend Completion: Wire Dead Kernels, Fused GEMM, I/O Manager

## TL;DR

> **Quick Summary**: Complete the Triton backend by wiring 9 dead Triton kernels into the tile pipeline, adding an optional fused Triton GEMM kernel, and connecting the I/O Manager — transforming the current "cuBLAS+PyTorch with Triton noise" into a genuine Triton-accelerated backend.
> 
> **Deliverables**:
> - Pulsed update pipeline wired through Triton kernels (ConstantStep tile; ExpStep stays PyTorch)
> - Optional fused Triton GEMM (matmul + noise + clamp in one kernel) via `use_triton_gemm` flag
> - TritonIOManager connected to tile forward/backward (gated by existing IOParameters config)
> - Kernel-level test suite (`tests/test_triton_kernels.py`)
> - Bug fixes: graceful fallback, duplicate Bernoulli code, Triton tile test parametrization
> - Integration with existing examples (01, 03, 04)
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: T1 (bug fixes) → T4 (pulsed pipeline) → T6 (I/O Manager) → T8 (fused GEMM) → T11 (kernel tests) → T13 (examples) → F1-F4

---

## Context

### Original Request
Code review of 4 local commits (26 files, 4237 lines) revealed that the "Triton backend" runs only 3 of 13 Triton kernels, delegates GEMM to cuBLAS, implements pulsed updates in pure PyTorch, and leaves the I/O Manager unwired. User requested a concrete completion plan to achieve genuine Triton acceleration.

### Previous Work (4 commits ahead of origin/master)
| Hash | Message | Files | Lines |
|------|---------|-------|-------|
| `2967083` | `feat(triton): add backend types, interfaces, RNG and math kernels` | 10 | +1059 |
| `fe7c3c8` | `feat(triton): add fused MVM forward/backward and I/O kernels` | 6 | +1249 |
| `36260c6` | `feat(triton): add device models and all Triton tile implementations` | 8 | +1464 |
| `4782d1c` | `feat(triton): add test suite and LeNet5 training example` | 2 | +465 |

### Interview Summary
**Key Decisions**:
- **torch.autograd.Function**: KEEP (unsloth uses same pattern, speed impact negligible vs kernel wiring)
- **Fused Triton GEMM**: Add as optional mode (`use_triton_gemm` flag), cuBLAS remains default
- **Scope**: Dead kernel wiring (#1) + Fused GEMM (#2) + I/O Manager (#3). triton_op migration excluded.

**Research Findings**:
- 13 @triton.jit kernels defined, only 3 are live (noise generation + noise/clamp fusion)
- 9 dead kernel paths: 7 never-called kernels + 2 wrapper-exists-but-tiles-skip
- unsloth: 22 Triton kernels, uses autograd.Function + custom_op (not triton_op)
- unsloth GEMM: autotuned BLOCK_SIZE_M/N/K (32-256), fused with permutation
- Forward/backward GEMM is ~90% of compute, currently on cuBLAS
- Pulsed update entirely in PyTorch (ConstantStep, ExpStep)

### Metis Review
**Identified Gaps** (addressed):
- Pulsed update pipeline needs glue code (scale → quantize → kernel), not just "call triton_pulsed_update" — task explicitly includes pipeline design
- I/O Manager must be gated behind config (changes forward computation semantics)
- ExpStep Triton kernel doesn't exist — stays on PyTorch fallback this iteration
- exp_step.py duplicate Bernoulli is correctness bug, not just code dup — fixed first
- TritonIOManager is pure Python internally — separate task to accelerate with Triton kernels
- Fused GEMM scoped to float32, [batch, in] × [out, in].T, no mixed precision
- Non-power-of-2 matrix dimension handling mandatory in fused GEMM

---

## Work Objectives

### Core Objective
Transform the aihwkit Triton backend from "cuBLAS GEMM + PyTorch update + Triton noise" into a genuine Triton-accelerated backend where forward, backward, update, and I/O operations use Triton kernels on the hot path.

### Concrete Deliverables
- Modified `src/aihwkit/simulator/triton/tiles/constant_step.py` — pulsed update via Triton kernel pipeline
- Modified `src/aihwkit/simulator/triton/tiles/analog.py` — I/O Manager integration + fused GEMM path
- New fused GEMM kernel in `src/aihwkit/simulator/triton/kernels/fused_gemm.py`
- Modified `src/aihwkit/simulator/triton/io_manager.py` — accelerated with Triton math kernels
- New `tests/test_triton_kernels.py` — dedicated kernel-level tests
- Modified `tests/helpers/tiles.py` — Triton tile variants in parametrization
- Modified examples (01, 03, 04) — Triton backend integration
- Bug fixes in `__init__.py`, `configs.py`, `exp_step.py`

### Definition of Done
- [x] `pytest tests/test_triton_tiles.py -q` → ALL 24+ PASS (no regression)
- [x] `pytest tests/test_triton_kernels.py -q` → ALL PASS (new kernel tests)
- [x] `pytest tests/test_simulator_tiles.py -k "Triton" -q` → ALL PASS (parametrized suite)
- [x] `AIHWKIT_USE_TRITON=1 python examples/04_lenet5_triton.py` → converges, accuracy > 95%
- [x] ConstantStepTritonTile.update() executes Triton pulsed_update kernel (not PyTorch path)
- [x] `from aihwkit.nn import AnalogLinear` works without Triton installed (graceful fallback)
- [x] Fused GEMM: `triton_fused_gemm(W, x)` matches `torch.mm(x, W.T)` within 1e-5

### Must Have
- Pulsed update pipeline: scale → quantize (bit_line_maker) → triton_pulsed_update for ConstantStep
- Fused Triton GEMM kernel (GEMM + noise + clamp in one kernel launch), gated by `use_triton_gemm`
- TritonIOManager wired into forward/backward, gated by IOParameters config
- TritonIOManager internals using Triton math kernels (abs_max, clamp, scale) on GPU
- Graceful Triton fallback (lazy import, no hard ImportError at package level)
- Fix duplicate Bernoulli bug in exp_step.py
- CPU fallback for every new public kernel function
- Kernel-level test suite
- Non-power-of-2 matrix dimension support in fused GEMM

### Must NOT Have (Guardrails)
- DO NOT modify `base_tile.py` abstract interface
- DO NOT change default behavior of any existing tile (fused GEMM and I/O Manager are opt-in)
- DO NOT write an ExpStep Triton kernel (stays on PyTorch fallback)
- DO NOT modify files outside `src/aihwkit/simulator/triton/`, `src/aihwkit/simulator/configs/`, `tests/`, `examples/`
- DO NOT touch CUDA/C++ backend (`src/rpucuda/`)
- DO NOT implement mixed precision in fused GEMM (float32 only)
- DO NOT make fused GEMM the default path (must be opt-in via flag)
- DO NOT over-abstract: no "generic kernel framework"
- DO NOT break the 24 existing tests in test_triton_tiles.py

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest + test_triton_tiles.py + helpers/tiles.py)
- **Automated tests**: Tests-after (extend existing + new kernel tests)
- **Framework**: pytest
- **Strategy**: New `test_triton_kernels.py` for kernel-level tests + Triton variants in parametrized suite

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Triton kernels**: Use Bash (Python REPL) — import kernel, create test tensors, compare with PyTorch reference
- **Tile operations**: Use Bash (pytest) — run specific test classes with Triton tiles
- **Training examples**: Use Bash (python) — run example scripts, verify loss convergence
- **Integration**: Use Bash (pytest) — run full test suite

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Cleanup & Foundation — immediate, no dependencies):
├── Task 1: Fix exp_step.py duplicate Bernoulli bug [quick]
├── Task 2: Fix graceful Triton fallback (__init__.py + configs.py) [quick]
├── Task 3: Add config flags (use_triton_gemm on SingleRPUConfig) [quick]

Wave 2 (Kernel Pipeline Wiring — core architectural work):
├── Task 4: Wire pulsed update pipeline for ConstantStepTritonTile [deep]
├── Task 5: Wire TritonIOManager into TritonAnalogTile forward/backward [deep]
├── Task 6: Accelerate TritonIOManager internals with Triton kernels [unspecified-high]

Wave 3 (Fused GEMM — highest risk, most isolated):
├── Task 7: Write fused Triton GEMM kernel (matmul + noise + clamp) [deep]
├── Task 8: Wire fused GEMM into forward/backward paths [unspecified-high]

Wave 4 (Testing & Integration):
├── Task 9: Create tests/test_triton_kernels.py [unspecified-high]
├── Task 10: Extend tests/helpers/tiles.py with Triton variants [unspecified-high]
├── Task 11: Install parameterized dep + verify test_simulator_tiles.py [quick]
├── Task 12: Integrate examples 01, 03, 04 with Triton paths [unspecified-high]
├── Task 13: Benchmark Triton vs cuBLAS forward/backward/update timing [unspecified-high]

Wave FINAL (Verification — independent reviews, 4 parallel):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
├── Task F4: Scope fidelity check (deep)

Critical Path: T1 → T4 → T5 → T7 → T9 → T12 → F1-F4
Parallel Speedup: ~55% faster than sequential
Max Concurrent: 3 (Waves 1 & 3)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 4 | 1 |
| 2 | — | all | 1 |
| 3 | — | 8 | 1 |
| 4 | 1, 2 | 9, 10 | 2 |
| 5 | 2 | 6, 9 | 2 |
| 6 | 5 | 9 | 2 |
| 7 | 2 | 8 | 3 |
| 8 | 3, 7 | 9, 12 | 3 |
| 9 | 4, 5, 6, 8 | F1-F4 | 4 |
| 10 | 4, 5 | 11 | 4 |
| 11 | 10 | F1-F4 | 4 |
| 12 | 4, 5, 8 | F1-F4 | 4 |
| 13 | 4, 5, 8 | F1-F4 | 4 |

### Agent Dispatch Summary

- **Wave 1**: 3 tasks — T1 → `quick`, T2 → `quick`, T3 → `quick`
- **Wave 2**: 3 tasks — T4 → `deep`, T5 → `deep`, T6 → `unspecified-high`
- **Wave 3**: 2 tasks — T7 → `deep`, T8 → `unspecified-high`
- **Wave 4**: 5 tasks — T9 → `unspecified-high`, T10 → `unspecified-high`, T11 → `quick`, T12 → `unspecified-high`, T13 → `unspecified-high`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.

---

### Wave 1: Cleanup & Foundation

- [x] 1. Fix exp_step.py Duplicate Bernoulli Bug

  **What to do**:
  - Open `src/aihwkit/simulator/triton/tiles/exp_step.py`
  - Lines 112-114 and 116-118 contain EXACT duplicate Bernoulli pulse generation
  - Remove lines 116-118 (the second duplicate block)
  - This is a correctness bug: double PRNG consumption, wasted computation

  **Must NOT do**:
  - Do NOT change the Bernoulli sampling logic — only remove the duplicate

  **Recommended Agent Profile**:
  - **Category**: `quick` — Single file, 3-line deletion
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 4 | **Blocked By**: None

  **References**:
  - `src/aihwkit/simulator/triton/tiles/exp_step.py:112-118` — Duplicate lines
  - `src/aihwkit/simulator/triton/tiles/constant_step.py:108-114` — Correct reference

  **Acceptance Criteria**:
  - [ ] `grep -c 'torch.bernoulli' src/aihwkit/simulator/triton/tiles/exp_step.py` → exactly 2 (not 4)
  - [ ] `pytest tests/test_triton_tiles.py::TestExpStepTritonTile -q` → ALL PASS

  **QA Scenarios:**
  ```
  Scenario: Duplicate removal
    Tool: Bash
    Steps:
      1. grep -c 'torch.bernoulli' src/aihwkit/simulator/triton/tiles/exp_step.py
      2. Assert output is '2'
      3. pytest tests/test_triton_tiles.py::TestExpStepTritonTile -q
    Expected Result: 2 Bernoulli calls, all tests pass
    Evidence: .sisyphus/evidence/task-1-bernoulli-fix.txt
  ```

  **Commit**: YES (Wave 1) | `fix(triton): fix duplicate Bernoulli bug and graceful fallback`

- [x] 2. Fix Graceful Triton Fallback

  **What to do**:
  - Modify `src/aihwkit/simulator/triton/__init__.py`: lazy import with `_triton_available` flag
    - Current: hard `import triton` → crash if not installed
    - New: `try: import triton; _triton_available = True except: _triton_available = False`
    - Expose `is_triton_available()` function
  - Modify `src/aihwkit/simulator/configs/configs.py:125,288`: wrap TritonBackend import in try/except
  - Ensure `from aihwkit.nn import AnalogLinear` works without Triton

  **Must NOT do**:
  - Do NOT make Triton a hard dependency

  **Recommended Agent Profile**:
  - **Category**: `quick` — Import guard pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 1 (with Tasks 1, 3)
  - **Blocks**: All subsequent tasks | **Blocked By**: None

  **References**:
  - `src/aihwkit/simulator/triton/__init__.py:9` — Hard ImportError
  - `src/aihwkit/simulator/configs/configs.py:125,288` — Inline imports that propagate errors

  **Acceptance Criteria**:
  - [ ] With Triton: `is_triton_available()` returns True
  - [ ] Without Triton: `from aihwkit.nn import AnalogLinear` succeeds
  - [ ] Without Triton: `SingleRPUConfig(use_triton=True)` falls back with warning

  **QA Scenarios:**
  ```
  Scenario: Graceful fallback
    Tool: Bash (python)
    Steps:
      1. python -c "import sys; sys.modules['triton']=None; from aihwkit.nn import AnalogLinear; print('OK')"
      2. Assert output contains 'OK'
    Expected Result: No crash, AnalogLinear imports
    Evidence: .sisyphus/evidence/task-2-graceful-fallback.txt
  ```

  **Commit**: YES (Wave 1) | `fix(triton): fix duplicate Bernoulli bug and graceful fallback`

- [x] 3. Add Config Flag for Fused GEMM

  **What to do**:
  - Add `use_triton_gemm: bool = False` to `SingleRPUConfig` in configs.py
  - Plumb to tile: `TritonAnalogTile.__init__` reads and stores as `self._use_triton_gemm`
  - Default False → existing cuBLAS path unchanged

  **Must NOT do**: Do NOT implement fused GEMM (Task 7)

  **Recommended Agent Profile**: `quick` | **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 1 | **Blocks**: Task 8 | **Blocked By**: None

  **References**:
  - `src/aihwkit/simulator/configs/configs.py` — `use_triton` field pattern (line ~117)

  **Acceptance Criteria**:
  - [ ] `SingleRPUConfig(use_triton_gemm=True)` accepted
  - [ ] `SingleRPUConfig().use_triton_gemm == False`

  **QA Scenarios:**
  ```
  Scenario: Config flag
    Tool: Bash (python)
    Steps:
      1. python -c "from aihwkit.simulator.configs import SingleRPUConfig; print(SingleRPUConfig().use_triton_gemm)"
      2. Assert 'False'
    Expected Result: Default False
    Evidence: .sisyphus/evidence/task-3-config-flag.txt
  ```

  **Commit**: YES (Wave 1) | `fix(triton): fix duplicate Bernoulli bug and graceful fallback`

### Wave 2: Kernel Pipeline Wiring

- [x] 4. Wire Pulsed Update Pipeline for ConstantStepTritonTile

  **What to do**:
  - Rewrite `ConstantStepTritonTile.update()` to use the Triton kernel pipeline instead of pure PyTorch:
    1. Compute scale factors using CUDA reference formula: `base_A = sqrt(lr / (dw_min * BL))`
    2. Scale inputs: use `triton_row_max` (maximizer.py) for per-row absolute max, then `triton_scale_rows` for normalization
    3. Quantize to pulse counts: use `triton_get_counts` (bit_line_maker.py) to convert scaled values to int32 counts
    4. Generate signed pulse counts: `x_counts * x_sign`, `d_counts * (-d_sign)`
    5. Call `triton_pulsed_update(weights, x_counts, d_counts, dw_min, ...)` with CONSTANT_STEP functor
    6. Clamp weights to [w_min, w_max]
  - This replaces the current all-PyTorch Bernoulli + outer product approach with C++-reference-matching pulse count pipeline
  - Import and use: `triton_pulsed_update` from kernels/pulsed_update.py, `triton_get_counts` from kernels/bit_line_maker.py, `triton_row_max`/`triton_scale_rows` from kernels/maximizer.py
  - Handle batch dimension: reduce over batch before quantization (mean or per-sample processing)
  - Handle 3D inputs (conv unfold): flatten to 2D before processing
  - Handle edge cases: BL=0 (skip update), zero inputs

  **Must NOT do**:
  - Do NOT modify the Triton kernels themselves — they're already implemented
  - Do NOT write an ExpStep Triton kernel (ExpStep stays on current PyTorch path)
  - Do NOT change the ideal update path in TritonAnalogTile.update()

  **Recommended Agent Profile**:
  - **Category**: `deep` — Core architectural piece, must match C++ reference formula
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 2 (with Tasks 5, 6)
  - **Blocks**: Tasks 9, 10 | **Blocked By**: Tasks 1, 2

  **References**:
  - `src/aihwkit/simulator/triton/tiles/constant_step.py` — Current PyTorch implementation to replace
  - `src/aihwkit/simulator/triton/kernels/pulsed_update.py` — `triton_pulsed_update()` wrapper (lines ~300-350)
  - `src/aihwkit/simulator/triton/kernels/bit_line_maker.py` — `triton_get_counts()` for pulse quantization
  - `src/aihwkit/simulator/triton/kernels/maximizer.py` — `triton_row_max()`, `triton_scale_rows()` for input scaling
  - `src/rpucuda/rpu_pulsed_meta_parameter.cpp` — C++ reference formula: `base_A = sqrt(lr / (dw_min * BL))`
  - `src/rpucuda/rpu_pulsed_meta_parameter.cpp` — C++ reference: CUDA pulsed update formula details (search for `base_A` or `dw_min`)

  **Acceptance Criteria**:
  - [ ] ConstantStepTritonTile.update() calls triton_pulsed_update (verify via debug print or breakpoint)
  - [ ] `pytest tests/test_triton_tiles.py::TestConstantStepTritonTile -q` → ALL PASS
  - [ ] Weights stay within [w_min, w_max] after 100 updates
  - [ ] Training with ConstantStep converges (loss decreases over epochs)

  **QA Scenarios:**
  ```
  Scenario: Pulsed update uses Triton kernel
    Tool: Bash (python)
    Steps:
      1. Create ConstantStepTritonTile, set weights to zero
      2. Call update(x, d) with non-zero inputs 10 times
      3. Assert weights changed (non-zero) AND stay within bounds
      4. pytest tests/test_triton_tiles.py::TestConstantStepTritonTile -q
    Expected Result: Weights updated via Triton kernel path, bounds enforced
    Evidence: .sisyphus/evidence/task-4-pulsed-pipeline.txt

  Scenario: Training convergence with Triton pulsed update
    Tool: Bash (python)
    Steps:
      1. AIHWKIT_USE_TRITON=1 python -c "from aihwkit.nn import AnalogLinear; ..." (simple training loop)
      2. Assert loss decreases over 10 steps
    Expected Result: Loss convergence
    Evidence: .sisyphus/evidence/task-4-training-convergence.txt
  ```

  **Commit**: YES (Wave 2) | `feat(triton): wire pulsed update pipeline and I/O manager`

- [x] 5. Wire TritonIOManager into TritonAnalogTile Forward/Backward

  **What to do**:
  - Modify `TritonAnalogTile._forward_impl()` to use TritonIOManager for input/output management:
    - Before GEMM: `io_result = self._io_manager.manage_input(x, self._f_io)`
    - GEMM: `y = forward_mvm(weights, io_result.x_scaled, ...)`
    - After GEMM: `y = self._io_manager.manage_output(y, self._f_io, io_result.scale)`
  - Similarly modify `_backward_impl()` for backward I/O management
  - **Gate by IOParameters config**: Only use I/O Manager when `io_pars.noise_management != 'NONE'` or `io_pars.bound_management != 'NONE'` (i.e., when I/O management is actually configured)
  - When IOParameters has default values (no noise, no bound management), the existing direct path is used (preserving backward compat)
  - Initialize `self._io_manager = TritonIOManager()` in `__init__`
  - Import TritonIOManager in analog.py

  **Must NOT do**:
  - Do NOT change behavior when IOParameters has default/none settings
  - Do NOT modify TritonIOManager class itself (that's Task 6)

  **Recommended Agent Profile**:
  - **Category**: `deep` — Changes core forward/backward path, must preserve backward compatibility
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 2 (with Tasks 4, 6)
  - **Blocks**: Task 6, 9 | **Blocked By**: Task 2

  **References**:
  - `src/aihwkit/simulator/triton/tiles/analog.py:160-184` — Current _forward_impl (reads out_noise, out_bound directly)
  - `src/aihwkit/simulator/triton/io_manager.py` — TritonIOManager with manage_input/manage_output
  - `src/aihwkit/simulator/parameters/io.py:IOParameters` — I/O config fields (inp_bound, out_bound, noise_management, etc.)
  - `src/aihwkit/simulator/tiles/torch_tile.py` — TorchTile I/O handling as golden reference

  **Acceptance Criteria**:
  - [ ] With default IOParameters: forward output unchanged (no regression)
  - [ ] With noise_management=ABS_MAX: input is scaled before GEMM
  - [ ] `pytest tests/test_triton_tiles.py -q` → ALL 24 PASS (no regression)

  **QA Scenarios:**
  ```
  Scenario: No regression with default I/O
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/test_triton_tiles.py -q
      2. Assert all 24 tests pass
    Expected Result: No regression
    Evidence: .sisyphus/evidence/task-5-io-no-regression.txt

  Scenario: I/O management with ABS_MAX scaling
    Tool: Bash (python)
    Steps:
      1. Create tile with IOParameters(noise_management='ABS_MAX', inp_bound=1.0)
      2. Forward with large input (values > 1.0)
      3. Assert output is bounded/scaled appropriately
    Expected Result: Input scaled, output within bounds
    Evidence: .sisyphus/evidence/task-5-io-abs-max.txt
  ```

  **Commit**: YES (Wave 2) | `feat(triton): wire pulsed update pipeline and I/O manager`

- [x] 6. Accelerate TritonIOManager Internals with Triton Kernels

  **What to do**:
  - Modify `src/aihwkit/simulator/triton/io_manager.py` to use Triton kernels on GPU:
    - Replace `x.abs().max().item()` with `triton_abs_max(x)` from math_utils.py (avoids CPU sync)
    - Replace `torch.clamp(x, ...)` with `triton_clamp(x, ...)` from math_utils.py
    - Replace manual per-row scaling with `triton_row_max` + `triton_scale_rows` from maximizer.py
    - Replace `x * scale` with `triton_elem_scale(x, scale)` from math_utils.py
  - Keep CPU fallback: check `x.is_cuda` before using Triton kernels, fall back to PyTorch otherwise
  - This turns 7 dead kernels into live code (abs_max, clamp, elem_scale, row_max, scale_rows)

  **Must NOT do**:
  - Do NOT change TritonIOManager's public API (manage_input, manage_output signatures)
  - Do NOT break CPU fallback

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — Multiple kernel integrations, must maintain CPU fallback
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 5 for wiring)
  - **Blocks**: Task 9 | **Blocked By**: Task 5

  **References**:
  - `src/aihwkit/simulator/triton/io_manager.py` — Current pure-Python implementation
  - `src/aihwkit/simulator/triton/kernels/math_utils.py` — triton_abs_max, triton_clamp, triton_elem_scale
  - `src/aihwkit/simulator/triton/kernels/maximizer.py` — triton_row_max, triton_scale_rows

  **Acceptance Criteria**:
  - [ ] On GPU: I/O Manager uses Triton kernels (no `.item()` CPU sync calls)
  - [ ] On CPU: I/O Manager falls back to PyTorch ops
  - [ ] `pytest tests/test_triton_tiles.py -q` → ALL PASS

  **QA Scenarios:**
  ```
  Scenario: GPU path uses Triton kernels
    Tool: Bash (python)
    Steps:
      1. Create TritonIOManager, call manage_input with CUDA tensor
      2. Verify no CPU sync points (no .item() calls in hot path)
    Expected Result: Triton kernels used on GPU
    Evidence: .sisyphus/evidence/task-6-io-triton-accel.txt
  ```

  **Commit**: YES (Wave 2) | `feat(triton): wire pulsed update pipeline and I/O manager`

### Wave 3: Fused GEMM

- [x] 7. Write Fused Triton GEMM Kernel

  **What to do**:
  - Create `src/aihwkit/simulator/triton/kernels/fused_gemm.py` with a fused GEMM kernel:
  - Implement `@triton.jit` kernel `_fused_gemm_noise_clamp_kernel`:
    - Computes: `out = clamp(x @ W.T + noise * noise_std, -bound, bound)` in ONE kernel launch
    - Tiling: BLOCK_M (rows of x), BLOCK_N (cols of W.T = rows of W), BLOCK_K (reduction dim)
    - Accumulator: float32 (even if inputs are float32, for numerical stability)
    - Noise: generate inline using `tl.rand` + Box-Muller (or accept pre-generated noise pointer)
    - Clamp: fused into store step (`tl.store(tl.maximum(tl.minimum(acc, bound), -bound))`)
    - Masking: handle non-power-of-2 M, N, K with proper `tl.where` masks
  - `triton.autotune` with configs: BLOCK_M=[32,64,128], BLOCK_N=[32,64,128], BLOCK_K=[32,64], num_stages=[2,3], num_warps=[4,8]
  - Python wrapper `triton_fused_gemm(x, weights, noise_std, bound, seed)` that:
    - Handles 1D input (single vector) by unsqueezing to 2D
    - Handles 3D input (conv unfold) by reshaping to 2D
    - Has CPU fallback: `torch.mm(x, weights.T)` + noise + clamp
  - Adapt tiling pattern from unsloth `_grouped_gemm_forward_kernel` (strip MoE routing, add noise/clamp)
  - Similarly create `triton_fused_gemm_backward(d, weights, noise_std, bound, seed)` for transposed GEMM

  **Must NOT do**:
  - Do NOT support mixed precision (float32 only)
  - Do NOT implement persistent kernels or flash-attention patterns
  - Do NOT make this the default path (opt-in only)
  - Do NOT try to beat cuBLAS — goal is fusion (eliminate separate kernel launches for noise+clamp)

  **Recommended Agent Profile**:
  - **Category**: `deep` — Performance-critical kernel, tiling, autotuning, edge cases
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 3 (with Task 8 sequential)
  - **Blocks**: Task 8 | **Blocked By**: Task 2

  **References**:
  - `unsloth/kernels/moe/grouped_gemm/kernels/forward.py:_grouped_gemm_forward_kernel` — Triton GEMM tiling pattern
  - `unsloth/kernels/fp8.py:_w8a8_block_fp8_matmul` — Alternative GEMM with per-block scaling
  - Triton matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  - `src/aihwkit/simulator/triton/kernels/forward.py` — Current hybrid implementation to replace

  **Acceptance Criteria**:
  - [ ] `triton_fused_gemm(x, W, 0, inf, 0)` matches `torch.mm(x, W.T)` within 1e-5
  - [ ] Works for non-power-of-2 dims: (1,8,4), (16,256,128), (32,513,255)
  - [ ] With noise: output differs from clean GEMM by expected magnitude
  - [ ] CPU fallback produces correct results
  - [ ] 1D and 3D input handling correct

  **QA Scenarios:**
  ```
  Scenario: Fused GEMM numerical correctness
    Tool: Bash (python)
    Steps:
      1. python -c "
         import torch
         from aihwkit.simulator.triton.kernels.fused_gemm import triton_fused_gemm
         W = torch.randn(256, 128, device='cuda')
         x = torch.randn(32, 128, device='cuda')
         result = triton_fused_gemm(x, W, noise_std=0.0, bound=float('inf'), seed=0)
         expected = torch.mm(x, W.T)
         print(f'max_diff={(result-expected).abs().max().item()}')
         assert torch.allclose(result, expected, atol=1e-5)
         print('PASS')"
    Expected Result: Match within 1e-5
    Evidence: .sisyphus/evidence/task-7-fused-gemm-correctness.txt

  Scenario: Non-power-of-2 dimensions
    Tool: Bash (python)
    Steps:
      1. Test with shapes (32, 513) x (255, 513).T = (32, 255)
      2. Assert no crash, correct shape, matches torch.mm
    Expected Result: Correct handling of odd dimensions
    Evidence: .sisyphus/evidence/task-7-fused-gemm-nonpow2.txt
  ```

  **Commit**: YES (Wave 3) | `feat(triton): add fused Triton GEMM kernel with optional mode`

- [x] 8. Wire Fused GEMM into Forward/Backward Paths

  **What to do**:
  - Modify `TritonAnalogTile._forward_impl()` in analog.py:
    - When `self._use_triton_gemm is True`: call `triton_fused_gemm(x, weights, noise_std, bound, seed)`
    - When `False` (default): keep existing `triton_forward_mvm()` path (cuBLAS + separate noise/clamp)
  - Similarly modify `_backward_impl()` for backward path
  - Import `triton_fused_gemm` and `triton_fused_gemm_backward` from kernels/fused_gemm.py
  - All subclasses (ConstantStep, ExpStep, Inference) inherit fused GEMM support automatically

  **Must NOT do**:
  - Do NOT change the default code path (use_triton_gemm=False stays as-is)
  - Do NOT remove the existing triton_forward_mvm/backward_mvm functions

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — Conditional path in forward/backward, must preserve backward compat
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 7)
  - **Blocks**: Tasks 9, 12 | **Blocked By**: Tasks 3, 7

  **References**:
  - `src/aihwkit/simulator/triton/tiles/analog.py:_forward_impl` — Where to add conditional
  - `src/aihwkit/simulator/triton/kernels/fused_gemm.py` (Task 7 output)

  **Acceptance Criteria**:
  - [ ] `use_triton_gemm=False`: behavior identical to before (no regression)
  - [ ] `use_triton_gemm=True`: uses fused GEMM, produces correct results
  - [ ] `pytest tests/test_triton_tiles.py -q` → ALL PASS

  **QA Scenarios:**
  ```
  Scenario: No regression with default path
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/test_triton_tiles.py -q
      2. Assert all 24 tests pass
    Expected Result: No regression
    Evidence: .sisyphus/evidence/task-8-no-regression.txt

  Scenario: Fused GEMM forward path
    Tool: Bash (python)
    Steps:
      1. Create TritonAnalogTile with use_triton_gemm=True
      2. Run forward, verify output shape and values
    Expected Result: Correct output via fused GEMM
    Evidence: .sisyphus/evidence/task-8-fused-forward.txt
  ```

  **Commit**: YES (Wave 3) | `feat(triton): add fused Triton GEMM kernel with optional mode`

### Wave 4: Testing & Integration

- [x] 9. Create tests/test_triton_kernels.py

  **What to do**:
  - Create `tests/test_triton_kernels.py` with dedicated kernel-level unit tests:
  - Test each public kernel wrapper (not internal `_kernel` functions):
    - `triton_abs_max`: compare with `torch.abs(x).max()` for random tensors (various sizes)
    - `triton_clamp`: compare with `torch.clamp()`, test boundary values
    - `triton_elem_scale`: compare with `x * scalar`
    - `triton_row_max`: compare with `x.abs().max(dim=1).values`
    - `triton_scale_rows`: verify output rows have max abs value = 1.0
    - `triton_get_counts`: verify deterministic rounding for known inputs
    - `triton_pulsed_update`: verify coincidence detection (only x!=0 AND d!=0 positions update)
    - `triton_fused_gemm`: compare with `torch.mm()` for various matrix sizes including non-power-of-2
    - `triton_fused_gemm_backward`: compare with transposed GEMM
  - Add `@pytest.mark.skipif(not torch.cuda.is_available())` to all GPU tests
  - Add `@pytest.mark.skipif` for Triton availability
  - 1-3 tests per wrapper, ~15-20 total tests
  - Test both CUDA and CPU fallback paths

  **Must NOT do**:
  - Do NOT test internal `_kernel` functions directly (only public wrappers)
  - Do NOT exceed 25 tests (keep focused)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — Multiple test functions, various assertion patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 4 (with Tasks 10-13)
  - **Blocks**: F1-F4 | **Blocked By**: Tasks 4, 5, 6, 8

  **References**:
  - `tests/test_triton_tiles.py` — Existing test structure to follow
  - `src/aihwkit/simulator/triton/kernels/*.py` — All kernel wrappers to test

  **Acceptance Criteria**:
  - [ ] `pytest tests/test_triton_kernels.py -v` → 15+ tests ALL PASS
  - [ ] CPU fallback tests included and passing

  **QA Scenarios:**
  ```
  Scenario: Kernel tests pass
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/test_triton_kernels.py -v
      2. Assert all tests pass, count >= 15
    Expected Result: All kernel tests pass
    Evidence: .sisyphus/evidence/task-9-kernel-tests.txt
  ```

  **Commit**: YES (Wave 4) | `test(triton): add kernel tests, parametrized suite, examples`

- [x] 10. Extend tests/helpers/tiles.py with Triton Variants

  **What to do**:
  - Add Triton tile variants to `tests/helpers/tiles.py` parametrization:
    - `FloatingPointTriton`, `IdealTriton`, `ConstantStepTriton`, `ExpStepTriton`
  - Add `@pytest.mark.skipif(not triton_available)` guard
  - Follow existing pattern in tiles.py for tile type registration
  - Ensure existing parametrized tests (test_simulator_tiles.py) automatically include Triton variants

  **Recommended Agent Profile**: `unspecified-high` | **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 4 | **Blocks**: Task 11 | **Blocked By**: Tasks 4, 5

  **References**:
  - `tests/helpers/tiles.py` — Existing parametrization (935 lines, no Triton variants currently)
  - `tests/helpers/testcases.py` — `SKIP_CUDA_TESTS` pattern for skip logic

  **Acceptance Criteria**:
  - [ ] `pytest tests/test_simulator_tiles.py -k 'Triton' --collect-only` shows Triton test variants

  **Commit**: YES (Wave 4) | `test(triton): add kernel tests, parametrized suite, examples`

- [x] 11. Install parameterized Dep + Verify test_simulator_tiles.py

  **What to do**:
  - Install `parameterized` package: `pip install parameterized`
  - Run `pytest tests/test_simulator_tiles.py -k 'Triton' -q`
  - Fix any collection errors or failures

  **Recommended Agent Profile**: `quick` | **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO | **Blocked By**: Task 10

  **Acceptance Criteria**:
  - [ ] `pytest tests/test_simulator_tiles.py -k 'Triton' -q` → ALL PASS

  **Commit**: YES (Wave 4) | `test(triton): add kernel tests, parametrized suite, examples`

- [x] 12. Integrate Existing Examples with Triton Backend

  **What to do**:
  - Add Triton backend support to existing examples:
    - `examples/01_simple_layer.py`: Add `use_triton=True` variant or env var check
    - `examples/03_mnist_training.py`: Add Triton backend option
    - `examples/04_lenet5_training.py`: Add Triton backend option (separate from 04_lenet5_triton.py)
  - Ensure all 3 converge with Triton backend
  - Test with both `use_triton_gemm=False` (cuBLAS, default) and `use_triton_gemm=True` (fused)

  **Recommended Agent Profile**: `unspecified-high` | **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 4 | **Blocked By**: Tasks 4, 5, 8

  **Acceptance Criteria**:
  - [ ] `AIHWKIT_USE_TRITON=1 python examples/01_simple_layer.py` → loss converges
  - [ ] `AIHWKIT_USE_TRITON=1 python examples/03_mnist_training.py` → accuracy > 95%
  - [ ] `AIHWKIT_USE_TRITON=1 python examples/04_lenet5_training.py` → accuracy > 95%

  **Commit**: YES (Wave 4) | `test(triton): add kernel tests, parametrized suite, examples`

- [x] 13. Benchmark Triton vs cuBLAS Timing

  **What to do**:
  - Create a simple benchmark script that measures:
    - Forward pass timing: cuBLAS (default) vs fused GEMM (use_triton_gemm=True)
    - Pulsed update timing: PyTorch (old) vs Triton kernel pipeline (new)
    - Full training step timing: end-to-end comparison
  - Matrix sizes: 64x64, 256x256, 512x512, 1024x1024
  - Report: kernel launch count, wall time, speedup ratio
  - Save as `.sisyphus/evidence/benchmark-results.txt`

  **Recommended Agent Profile**: `unspecified-high` | **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES | Wave 4 | **Blocked By**: Tasks 4, 5, 8

  **Acceptance Criteria**:
  - [ ] Benchmark results saved to evidence directory
  - [ ] Report includes timing for all 4 matrix sizes

  **Commit**: YES (Wave 4) | `test(triton): add kernel tests, parametrized suite, examples`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run test, check import). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run linter + `pytest tests/ -k "Triton"`. Review all changed files for: empty excepts, print statements in prod, commented-out code, unused imports. Check all Triton kernels have CPU fallbacks. Verify no `base_tile.py` modifications. Check for duplicate code patterns.
  Output: `Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Run ALL training examples with Triton backend. Verify loss convergence. Run `pytest tests/test_triton_tiles.py` and `pytest tests/test_triton_kernels.py`. Test edge cases: empty tensor, single-element tensor, very large matrix, non-power-of-2 dimensions. Test fused GEMM vs cuBLAS numerical equivalence. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Examples [3/3 pass] | Tests [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built, nothing beyond spec. Check "Must NOT do" compliance. Verify no CUDA backend files modified. Verify base_tile.py unchanged. Flag files outside allowed paths.
  Output: `Tasks [N/N compliant] | CUDA Untouched [YES/NO] | Scope [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

| Wave | Commit | Message | Files | Pre-commit |
|------|--------|---------|-------|------------|
| 1 | 1 | `fix(triton): fix duplicate Bernoulli bug and graceful fallback` | exp_step.py, __init__.py, configs.py | `pytest tests/test_triton_tiles.py -x` |
| 2 | 2 | `feat(triton): wire pulsed update pipeline and I/O manager` | constant_step.py, analog.py, io_manager.py | `pytest tests/test_triton_tiles.py -x` |
| 3 | 3 | `feat(triton): add fused Triton GEMM kernel with optional mode` | fused_gemm.py, forward.py, backward.py | `pytest tests/test_triton_tiles.py -x` |
| 4 | 4 | `test(triton): add kernel tests, parametrized suite, examples` | tests/**, examples/** | `pytest tests/test_triton_kernels.py -x` |
| FINAL | 5 | `test(triton): verify full backend completion` | .sisyphus/evidence/** | `pytest tests/ -k "Triton"` |

---

## Success Criteria

### Verification Commands
```bash
# All existing Triton tile tests pass (no regression)
pytest tests/test_triton_tiles.py -q  # Expected: 24+ passed

# New kernel tests pass
pytest tests/test_triton_kernels.py -q  # Expected: 15+ passed

# Parametrized suite with Triton variants
pytest tests/test_simulator_tiles.py -k "Triton" -q  # Expected: ALL PASS

# Training examples converge
AIHWKIT_USE_TRITON=1 python examples/04_lenet5_triton.py  # Expected: accuracy > 95%

# Graceful fallback
python -c "from aihwkit.nn import AnalogLinear; print('OK')"  # Expected: OK (no crash without triton)

# CUDA backend unmodified
git diff src/rpucuda/  # Expected: no changes
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] All existing tests pass (no regression)
- [x] Fused GEMM matches cuBLAS within 1e-5
- [x] Pulsed update uses Triton kernel path
- [x] I/O Manager connected and gated
- [x] 3 training examples converge
- [x] CUDA backend files unmodified
- [x] base_tile.py unmodified

<!-- PLAN_READY_FOR_EXECUTION -->
