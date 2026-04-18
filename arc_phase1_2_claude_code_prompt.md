# ARC-AGI Neurogolf — Phase 1 & 2 Implementation

## Context

This is a Kaggle competition (Neurogolf) where the goal is to build the *smallest possible* ONNX
neural networks that reproduce ARC-AGI transformations. Scoring penalizes parameter count, memory
footprint, and multiply-accumulate operations. Smaller = better, as long as the network is
functionally correct across train/test/arc-gen splits and a private holdout set.

Each task lives in `data/tasks/task{NNN}.json` (001–400). Each file has:
- `"train"`: list of {input, output} grid pairs
- `"test"`: list of {input, output} grid pairs  
- `"arc-gen"`: list of {input, output} grid pairs

Grids are list-of-lists of ints 0–9. Before being fed to networks they are one-hot encoded into
tensors of shape [1, 10, 30, 30] with zero-padding outside the original grid border.

This session implements **Phase 1** (data pipeline + task classification) and **Phase 2**
(analytical ONNX network generators for identified transformation families). Do NOT implement RL or
gradient-based search — that is Phase 3.

---

## Project Structure to Create

```
arc_neurogolf/
├── data/
│   └── tasks/               # task001.json ... task400.json go here (user supplies)
├── pipeline/
│   ├── __init__.py
│   ├── loader.py            # JSON loading + grid encoding
│   ├── validator.py         # ONNX correctness checker + cost calculator
│   └── visualizer.py        # Optional: ASCII grid visualizer for debugging
├── classifier/
│   ├── __init__.py
│   ├── features.py          # Feature extraction per task
│   ├── rules.py             # Rule-based family classifier
│   └── families.py          # Enum + documentation of all known families
├── generators/
│   ├── __init__.py
│   ├── base.py              # Abstract base class for network generators
│   ├── color_remap.py       # 1x1 conv color permutation networks
│   ├── geometric.py         # Rot90, flip, translate networks
│   ├── local_rule.py        # 3x3 conv + threshold (Game-of-Life style)
│   ├── tiling.py            # Repeat/tile/upscale networks
│   └── identity.py          # Passthrough (sanity check baseline)
├── search/
│   └── __init__.py          # Placeholder for Phase 3
├── run_phase1.py            # CLI: classify all tasks, output report
├── run_phase2.py            # CLI: generate ONNX for classifiable tasks
├── requirements.txt
└── README.md
```

---

## Detailed Implementation Instructions

### 1. `requirements.txt`

```
numpy
onnx>=1.15.0
onnxruntime>=1.17.0
onnxscript
torch
torchvision
scipy
tqdm
rich
```

### 2. `pipeline/loader.py`

Implement the following functions:

**`load_task(task_id: int) -> dict`**
- Load `data/tasks/task{task_id:03d}.json`
- Return dict with keys `train`, `test`, `arc_gen` (rename `arc-gen`)
- Each value is list of dicts with `input` and `output` keys

**`encode_grid(grid: list[list[int]]) -> np.ndarray`**
- Input: variable-size grid (list of lists of ints 0–9)
- Output: float32 numpy array of shape [1, 10, 30, 30]
- One-hot encode each cell: channel `c` is 1.0 if pixel == c, else 0.0
- Cells outside the original grid border are zero across all channels
- The grid is placed top-left (row 0, col 0)

**`decode_grid(tensor: np.ndarray, height: int, width: int) -> list[list[int]]`**
- Input: float32 array shape [1, 10, 30, 30], original grid dimensions
- Output: list of lists of ints — argmax over channel dim for each cell
- Only return the [0:height, 0:width] region

**`get_all_pairs(task: dict) -> list[tuple]`**
- Returns all (input_grid, output_grid) pairs from train + test + arc_gen combined
- Each element is a tuple of (raw_input_grid, raw_output_grid)

### 3. `pipeline/validator.py`

**`run_network(onnx_path: str, input_tensor: np.ndarray) -> np.ndarray`**
- Run ONNX model via onnxruntime
- Return output tensor

**`check_correctness(onnx_path: str, task: dict) -> dict`**
- Run network on all pairs in train + test + arc_gen
- For each pair: decode output, compare to expected output grid exactly
- Return: `{correct: bool, n_pairs: int, n_correct: int, failures: list[int]}`
- A pair is correct only if ALL cells in the output region match exactly

**`compute_cost(onnx_path: str, input_shape=(1,10,30,30)) -> dict`**
- Compute:
  - `n_params`: total number of scalar parameters in initializers
  - `memory_bytes`: total bytes of all parameters (float32 = 4 bytes each; handle other dtypes)
  - `mac_ops`: multiply-accumulate operations — implement this for Conv, Gemm, MatMul ops
    by parsing the ONNX graph manually (do not use an external profiler)
  - `cost`: sum of all three
  - `score`: max(1, 25 - ln(cost))
- Return dict with all fields

**`validate_constraints(onnx_path: str) -> dict`**
- Check: all tensor shapes are static (no dynamic dims)
- Check: no forbidden ops (Loop, Scan, NonZero, Unique, Script, Function)
- Check: file size <= 1.44MB
- Return: `{valid: bool, violations: list[str]}`

### 4. `classifier/features.py`

**`extract_features(task: dict) -> dict`**

Extract the following features from all train pairs. Return a flat dict of scalar/bool features:

```python
{
  # Shape features
  "output_shape_eq_input": bool,       # all pairs have same H,W in/out
  "output_always_square": bool,
  "input_always_square": bool,
  "output_h_eq_input_w": bool,         # transposition hint
  "output_is_input_scaled": bool,      # output H/W is integer multiple of input
  "scale_factor_h": float,             # ratio if scaled, else 1.0
  "scale_factor_w": float,

  # Color/channel features
  "same_color_set": bool,              # same set of colors appears in/out
  "color_count_preserved": bool,       # same total pixel count per color
  "is_color_permutation": bool,        # bijective color mapping consistent across pairs
  "color_permutation_map": dict,       # {in_color: out_color} or None
  "uses_only_two_colors": bool,
  "background_color": int,             # most common color in inputs

  # Spatial/geometric features  
  "is_rot90": bool,                    # output is 90° rotation of input
  "is_rot180": bool,
  "is_rot270": bool,
  "is_flip_h": bool,                   # horizontal flip
  "is_flip_v": bool,                   # vertical flip
  "is_transpose": bool,                # matrix transpose
  "is_identity": bool,                 # output == input

  # Local rule features
  "output_is_single_channel": bool,    # output uses only 1 color + background
  "pixel_change_fraction": float,      # avg fraction of pixels that change
  "max_local_context_needed": int,     # estimated neighborhood size (1, 3, 5, ...)

  # Grid property features
  "input_has_border": bool,            # outermost ring is uniform color
  "output_has_border": bool,
  "input_max_h": int,
  "input_max_w": int,
  "output_max_h": int,
  "output_max_w": int,
}
```

For geometric checks (is_rot90, is_flip_h, etc.), test consistency across ALL train pairs —
not just one. A feature is True only if it holds for every pair in the train set.

For `is_color_permutation`: iterate all train pairs, build a mapping from each input color to
output color, check it is consistent and bijective across all pairs.

### 5. `classifier/families.py`

Define an enum `TransformFamily` with at least these values and docstrings:

```python
class TransformFamily(Enum):
    IDENTITY         = "identity"         # output == input
    COLOR_REMAP      = "color_remap"      # bijective color permutation, no spatial change
    ROT90            = "rot90"            # 90° clockwise rotation
    ROT180           = "rot180"           # 180° rotation
    ROT270           = "rot270"           # 270° clockwise rotation  
    FLIP_H           = "flip_h"           # horizontal flip (left-right)
    FLIP_V           = "flip_v"           # vertical flip (up-down)
    TRANSPOSE        = "transpose"        # matrix transpose
    SCALE_UP         = "scale_up"         # integer upscale (nearest neighbor)
    LOCAL_RULE_3x3   = "local_rule_3x3"  # output depends on 3x3 neighborhood
    LOCAL_RULE_5x5   = "local_rule_5x5"  # output depends on 5x5 neighborhood
    UNKNOWN          = "unknown"          # could not classify
```

Also define `FAMILY_PRIORITY`: ordered list from most-specific to least-specific, used by the
classifier to resolve ambiguity (IDENTITY > COLOR_REMAP > geometric families > local rules > UNKNOWN).

### 6. `classifier/rules.py`

**`classify_task(features: dict) -> TransformFamily`**

Rule-based classifier using extracted features. Priority order matters — check in this order:

1. `IDENTITY` if `is_identity`
2. `ROT90` if `is_rot90`
3. `ROT180` if `is_rot180`
4. `ROT270` if `is_rot270`
5. `FLIP_H` if `is_flip_h`
6. `FLIP_V` if `is_flip_v`
7. `TRANSPOSE` if `is_transpose`
8. `COLOR_REMAP` if `is_color_permutation` and `output_shape_eq_input`
9. `SCALE_UP` if `output_is_input_scaled` and `scale_factor_h >= 2` and `scale_factor_w >= 2`
10. `LOCAL_RULE_3x3` if `output_shape_eq_input` and `max_local_context_needed <= 3`
11. `LOCAL_RULE_5x5` if `output_shape_eq_input` and `max_local_context_needed <= 5`
12. `UNKNOWN` otherwise

**`classify_all_tasks(task_ids: list[int]) -> dict[int, TransformFamily]`**
- Run classify_task on each task id
- Return dict mapping task_id -> TransformFamily

### 7. `generators/base.py`

```python
from abc import ABC, abstractmethod
import onnx

class NetworkGenerator(ABC):
    family: str  # override in subclass

    @abstractmethod
    def can_generate(self, task: dict, features: dict) -> bool:
        """Return True if this generator can handle the given task."""

    @abstractmethod
    def generate(self, task: dict, features: dict) -> onnx.ModelProto:
        """Return a minimal ONNX model that solves this task."""

    def save(self, model: onnx.ModelProto, path: str):
        onnx.save(model, path)
```

### 8. `generators/identity.py`

Implement `IdentityGenerator(NetworkGenerator)`.

The identity network: output = input. Implement as a single ONNX `Identity` op applied to the
input tensor. No parameters. Verify this compiles and runs under onnxruntime.

### 9. `generators/color_remap.py`

Implement `ColorRemapGenerator(NetworkGenerator)`.

A color permutation is a 1×1 convolution with a 10×10 weight matrix W where:
- `W[out_color, in_color] = 1.0` if `color_map[in_color] == out_color`
- All other entries 0.0
- Bias = 0

This is a single `Conv` ONNX node with `kernel_shape=[1,1]`, `strides=[1,1]`, `pads=[0,0,0,0]`.
Weight tensor shape: [10, 10, 1, 1].

The `generate` method should:
1. Extract `color_permutation_map` from features
2. Build the 10×10 weight matrix
3. Create ONNX graph with one Conv node
4. Return the model

### 10. `generators/geometric.py`

Implement `GeometricGenerator(NetworkGenerator)` that handles ROT90, ROT180, ROT270, FLIP_H,
FLIP_V, TRANSPOSE.

For geometric transforms, use ONNX `Transpose` op (for permuting axes) and/or `Slice` + `Concat`
for flips. The key insight:

- **ROT90 CW**: in numpy terms, `np.rot90(grid, k=-1)` — implement via Transpose + Slice
- **FLIP_H** (left-right): reverse the W axis — implement via `Slice` with negative step or
  `Gather` with reversed index tensor as a constant initializer
- All flips and rotations can be expressed as combinations of `Transpose` and index reversal
- Use constant initializers for any index tensors needed
- Aim for the minimum number of ops — prefer a single Transpose where possible

For each sub-family, write a separate private `_build_{transform}(input_name) -> (nodes, inits)`
function that returns the ONNX nodes and initializers needed.

### 11. `generators/tiling.py`

Implement `TilingGenerator(NetworkGenerator)` for integer scale-up (nearest-neighbor).

A 2× upscale of a [1, 10, H, W] tensor to [1, 10, 2H, 2W] can be implemented with:
- ONNX `Resize` op with `mode='nearest'`, `coordinate_transformation_mode='asymmetric'`
- Scale factors: [1.0, 1.0, scale_h, scale_w] as a constant initializer

The `generate` method should read scale factors from features and build the Resize node.
Include the `roi` input (empty tensor) and `scales` input as required by opset 13+.

### 12. `generators/local_rule.py`

Implement `LocalRuleGenerator(NetworkGenerator)` for 3×3 neighborhood rules.

This is the hardest generator. The approach:
1. Run a brute-force search over 3×3 conv weights to find a weight matrix W such that
   `ReLU(Conv(input, W) - threshold) > 0` correctly predicts the output for all train pairs.
2. Use PyTorch to optimize W with Adam for up to 2000 steps.
3. After optimization, round weights to 2 decimal places and verify correctness.
4. If correct, export to ONNX using `torch.onnx.export`.
5. If not correct after optimization, return None (generator fails for this task).

The network architecture: `Conv(kernel=3, in_channels=10, out_channels=10, padding=1)` →
`ReLU` → `Conv(kernel=1, in_channels=10, out_channels=10)`. Keep this as the baseline
architecture; do not add more layers unless the first attempt fails twice.

Make the loss function:
- Binary cross-entropy between network output and one-hot target
- L1 regularization on weights (lambda=1e-4) to encourage sparsity

### 13. `run_phase1.py`

CLI script that:
1. Loads all available tasks from `data/tasks/`
2. Extracts features for each
3. Classifies each task
4. Prints a rich table summarizing:
   - Task ID | Family | Key Features (shape change, color perm, etc.)
5. Saves full results to `phase1_results.json`
6. Prints summary: count per family, number UNKNOWN

Usage: `python run_phase1.py`

### 14. `run_phase2.py`

CLI script that:
1. Loads `phase1_results.json`
2. For each non-UNKNOWN task, runs the appropriate generator
3. Validates the generated ONNX (correctness + constraints)
4. Saves valid ONNX files to `output/onnx/task{NNN}.onnx`
5. Computes and prints cost breakdown per task
6. Prints summary table: Task ID | Family | Cost | Score | Status
7. Saves full results to `phase2_results.json`

Usage: `python run_phase2.py`

---

## Implementation Notes & Constraints

- **ONNX opset**: use opset 17 throughout. Set this explicitly in all model builders.
- **Static shapes**: ALL tensor shapes in the ONNX graph must be statically defined. Never use
  symbolic dimensions. Input shape is always [1, 10, 30, 30]. Output shape must also be static.
  This means geometric generators that change shape (ROT90, SCALE_UP) must output a fixed shape
  — pad outputs back to [1, 10, 30, 30] with zeros and encode only the valid region.
- **Padding convention**: all outputs are [1, 10, 30, 30]. The valid grid region is top-left
  aligned; everything outside is zero across all channels.
- **No Python control flow in ONNX graphs**: no loops, no conditionals. Pure dataflow only.
- **float32 everywhere**: all parameters and activations should be float32 unless there is a
  specific reason otherwise.
- **Test every generator** with at least a synthetic unit test (not just on real tasks). Write
  these in a `tests/` directory.

---

## Validation After Each Generator

After implementing each generator, validate it end-to-end:
1. Find a real task of that family in the dataset (use phase1 classification)
2. Generate the ONNX file
3. Run `validate_constraints` — must pass
4. Run `check_correctness` on train + test + arc_gen — must be 100%
5. Run `compute_cost` — print the breakdown
6. If any step fails, fix the generator before moving on

Do not move to the next generator until the current one passes all checks.

---

## What NOT to do

- Do not implement RL, evolutionary search, or any gradient-based architecture search — that is Phase 3
- Do not use dynamic shapes anywhere in ONNX graphs
- Do not use forbidden ONNX ops: Loop, Scan, NonZero, Unique, Script, Function
- Do not hardcode task-specific weight values — generators must derive weights analytically from
  features or from a short PyTorch optimization (local_rule only)
- Do not skip the constraint validator — it must gate every ONNX file before it is saved

---

## Deliverables

When complete, the following should work:

```bash
python run_phase1.py        # classifies all 400 tasks, saves phase1_results.json
python run_phase2.py        # generates ONNX for all classifiable tasks, saves to output/onnx/
```

And the output directory should contain valid ONNX files for every task that was successfully
classified and generated.
