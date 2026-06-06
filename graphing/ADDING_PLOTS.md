# Adding plots to the graphing module

This document explains what data is available, how it is stored, and how to add new plots — either to the hard-coded standard set or as entity-specific soft-coded plots.

---

## Data schema

Everything the graphing module uses comes from MLflow. The pipeline logs one set of metrics per checkpoint step.

### Standard metrics (logged by `storage/mlflow_logger.py`)

These are logged for every run and every entity.

| MLflow metric key | Source field in `AnalysisResult` | Description |
|---|---|---|
| `opposition_score` | `opposition_score` | Cosine similarity between success/failure mean gradients. −1 = distinct, +1 = confused. |
| `coherence_success` | `coherence_success` | Average pairwise cosine similarity within success group gradient minibatches. |
| `coherence_failure` | `coherence_failure` | Average pairwise cosine similarity within failure group gradient minibatches. |
| `gradient_magnitude_success` | `gradient_magnitude_success` | L2 norm of the mean success gradient vector. |
| `gradient_magnitude_failure` | `gradient_magnitude_failure` | L2 norm of the mean failure gradient vector. |
| `activation_separation` | `activation_separation` | Euclidean distance between success/failure activation centroids. |
| `activation_cosine_distance` | `activation_cosine_distance` | Cosine distance (1 − cosine sim) between activation centroids. |
| `n_success` | `n_success` | Number of episodes in the success partition. |
| `n_failure` | `n_failure` | Number of episodes in the failure partition. |

### RSA alignment metrics

One metric per functional group defined in `entity.achievement_groups`. Group names come from the entity definition file (e.g. `entities/definitions/crafter.py`).

| MLflow metric key pattern | Description |
|---|---|
| `rsa_alignment_{group_name}` | Spearman ρ between model RDM and binary ground-truth matrix for group. |

For `ppo_crafter` the groups are: `fighting`, `resource`, `crafting`, `housing`.

### Achievement observation metrics

One binary metric per achievement, logged at each checkpoint.

| MLflow metric key pattern | Value | Description |
|---|---|---|
| `achievement_obs_{label_snake_case}` | 1.0 | Achievement was observed in evaluation at this checkpoint |
| `achievement_obs_{label_snake_case}` | 0.0 | Achievement was not observed |

Label is the display label from `entity.achievement_label_map` converted to lowercase snake_case (e.g. `"Eat Plant"` → `achievement_obs_eat_plant`).

These metrics are used by `graphing/loader.py` to populate `RunData.achievement_first_steps`.

### RDM artifacts

Stored in MLflow as JSON artifacts under the `rdm/` path.

| Artifact path | Format |
|---|---|
| `rdm/rdm_step{N}.json` | `{"rdm": [[float, ...], ...], "labels": ["label", ...], "step": N}` |

One artifact per checkpoint where RSA analysis was run. Use `loader.load_rdm_artifact(run_id)` to load the last one.

---

## `RunData` API

`graphing/loader.py` exposes a single `load_run()` function that returns a `RunData` dataclass:

```python
from graphing.loader import load_run

run_data = load_run("ppo_crafter")               # latest run
run_data = load_run("ppo_crafter", run_id="abc") # specific run
```

### `RunData` fields

| Field | Type | Description |
|---|---|---|
| `entity_id` | `str` | MLflow experiment name (= entity ID) |
| `run_id` | `str` | Full MLflow run ID |
| `params` | `dict[str, str]` | Run-level params (n_episodes, seed, percentile_x, ...) |
| `steps` | `list[int]` | Sorted checkpoint steps that have at least one logged metric |
| `metrics` | `dict[str, list[float]]` | Metric name → value per step (NaN where absent) |
| `rsa_groups` | `list[str]` | RSA group names found in this run |
| `achievement_first_steps` | `dict[str, int]` | Display label → first checkpoint step where obs > 0 |

Metric lists are aligned to `steps`: `metrics["opposition_score"][i]` corresponds to `steps[i]`.

### Helper

```python
from graphing.loader import load_rdm_artifact
rdm_data = load_rdm_artifact(run_id)   # last checkpoint
rdm_data = load_rdm_artifact(run_id, step=500000)
# Returns {"rdm": ..., "labels": ..., "step": ...} or None
```

---

## Standard metrics reference

| Metric | What it means | Range |
|---|---|---|
| `opposition_score` | Are success/failure gradients pointing in different directions? Low (negative) = better credit assignment. | −1 to +1 |
| `coherence_success/failure` | Do episodes within each group produce consistent gradients? High = agent reliably knows what to learn. | −1 to +1 |
| `gradient_magnitude_*` | How large are the gradients? Collapses near zero = vanishing signal. | ≥ 0 |
| `activation_separation` | Are the latent representations of good/bad episodes spatially distinct? | ≥ 0 |
| `activation_cosine_distance` | Cosine distance version of separation; scale-invariant. | 0 to 2 |
| `rsa_alignment_{group}` | Does the geometry of activation space reflect functional achievement structure? Positive = aligned. | −1 to +1 |

---

## Adding a standard plot

Standard plots live in `graphing/standard_plots.py` and are called for every entity by `graphing/run.py`. They must work without any entity-specific knowledge (no imports from `entities/`).

1. Add a function with this signature:

```python
def plot_my_metric(run_data: RunData, output_dir: str, dpi: int = 150) -> str | None:
    """Short description."""
    x = np.array(run_data.steps, dtype=float)
    y = np.array(run_data.metrics.get("my_metric_key", []), dtype=float)
    if not np.isfinite(y).any():
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    from graphing.styles import shaded_line, apply_base_style, fmt_millions, save_fig
    shaded_line(ax, x, y, color="#336699", label="My metric")
    ax.set_ylabel("Y axis label")
    ax.set_xlabel("Training step")
    fmt_millions(ax)
    apply_base_style(ax)
    ax.legend(fontsize=11)
    return save_fig(fig, os.path.join(output_dir, "my_metric.pdf"), dpi)
```

2. Add it to `plot_all()`:

```python
def plot_all(run_data, output_dir, dpi=150):
    ...
    p = plot_my_metric(run_data, output_dir, dpi)
    if p:
        saved.append(p)
    ...
```

---

## Adding an entity-specific plot

Entity plots live in `graphing/entities/{entity_id}.py`. They have access to the full `RunData` and can use entity-specific knowledge (achievement groups, label maps, etc.).

`graphing/run.py` automatically discovers and calls these files when `--entity` matches.

### Step 1 — create the file

```
graphing/entities/{entity_id}.py
```

### Step 2 — implement `plot()`

```python
"""Entity-specific plots for {entity_id}."""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np

from graphing.loader import RunData
from graphing.styles import apply_base_style, save_fig


def plot(run_data: RunData, output_dir: str, dpi: int = 150) -> None:
    """Entry point called by graphing/run.py. Generate all entity-specific plots."""
    _my_custom_plot(run_data, output_dir, dpi)


def _my_custom_plot(run_data: RunData, output_dir: str, dpi: int) -> str | None:
    # Use run_data.steps, run_data.metrics, run_data.achievement_first_steps, etc.
    ...
    return save_fig(fig, os.path.join(output_dir, "my_plot.pdf"), dpi)
```

### Step 3 — verify

```bash
python -m graphing.run --entity {entity_id} --output_dir graphs/
```

The `--- Entity-specific plots ---` section of the output should show your new file being saved.

---

## Adding a zoomed plot for a new metric

`graphing/zoomed_plots.py` contains `DEFAULT_ZOOM_METRICS` — the list of metrics for which zoomed plots are generated by default. To add a new metric:

1. Make sure the metric is logged to MLflow (it must appear in `run_data.metrics`).

2. Add the metric key to `DEFAULT_ZOOM_METRICS` in `graphing/zoomed_plots.py`:

```python
DEFAULT_ZOOM_METRICS = [
    "opposition_score",
    ...
    "my_new_metric",   # add here
]
```

3. Optionally add a human-readable Y-axis label to `_YLABEL`:

```python
_YLABEL: dict[str, str] = {
    ...
    "my_new_metric": "My new metric (units)",
}
```

Or call `plot_zoomed()` directly for full control:

```python
from graphing.zoomed_plots import plot_zoomed
plot_zoomed(run_data, metric="my_new_metric", window_checkpoints=8, output_dir="graphs/")
```
