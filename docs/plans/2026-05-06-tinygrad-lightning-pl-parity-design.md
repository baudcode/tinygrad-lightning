# tinygrad-lightning ↔ pytorch-lightning parity — design

**Status:** approved 2026-05-06
**Goal:** reimplement enough of `lightning.pytorch` on top of tinygrad ≥ 0.12 that a user can write a training script with the PL API surface, with a parity test suite that exercises the same features through both frameworks.

## Scope

In scope (this design):
- `LightningModule`, `Trainer.fit/validate/test`, `Callback`, `Logger`, `LightningDataModule`, `DataLoader`.
- Callbacks: `ModelCheckpoint`, `EarlyStopping`, `LearningRateMonitor`, `TQDMProgressBar`.
- Loggers: `MLFlowLogger` (optional), `TensorBoardLogger`, `CSVLogger`.
- LR schedulers: `StepLR`, `MultiStepLR`, `CosineAnnealingLR`, `LambdaLR`, `OneCycleLR`.
- Per-module learning rates via param groups in `configure_optimizers`.
- Single-device training on tinygrad's available accelerators.
- Mixed precision: `16-mixed`, `bf16-mixed`.
- Gradient clipping, `accumulate_grad_batches`, `seed_everything`.
- `self.log` with on_step/on_epoch reduction.

Out of scope (future):
- Distributed strategies (DDP, FSDP, DeepSpeed).
- The full PL hook surface beyond the ones above (`on_train_batch_end`, `on_before_optimizer_step`, etc. — added per-need).
- `predict_step`, `Trainer.predict`.
- `ReduceLROnPlateau` (deferred; needs `monitor` plumbing).
- Loading PyTorch `.ckpt` files.
- Fault tolerance / signal handling.

## Architecture

```
tinygrad_lightning/
  __init__.py              # public re-exports
  module.py                # LightningModule
  trainer/
    trainer.py             # Trainer + fit/validate/test loops
    states.py              # TrainerFn, RunningStage enums
    connectors.py          # callback/logger/optimizer connectors
  core/
    optimizer.py           # LightningOptimizer, param-group support
    lr_scheduler.py        # base + Step/MultiStep/Cosine/Lambda/OneCycle
    precision.py           # PrecisionPlugin (fp32, 16-mixed, bf16-mixed)
    accelerator.py         # CPU/GPU resolution to tinygrad device names
  callbacks/
    base.py                # Callback (subset of PL hook surface)
    checkpoint.py          # ModelCheckpoint
    early_stopping.py
    lr_monitor.py
    progress.py            # TQDMProgressBar
  loggers/
    base.py                # Logger ABC
    mlflow.py              # MLFlowLogger (lazy import)
    tensorboard.py
    csv.py
  data/
    dataloader.py
    datamodule.py
  utilities/
    seed.py                # seed_everything
    rank_zero.py
    types.py
tests/
  conftest.py              # parity_backend fixture, PL-skip marker
  parity/                  # behavioral parity (B-tier)
  numeric/                 # seed-pinned regressions (A-tier)
  unit/                    # standalone tinygrad-lightning tests
```

The new code replaces the existing 4-file structure in one shot. The codebase is small (~700 LOC) and there is no external user contract worth deprecating gradually.

## Public API surface

The exported names match `lightning.pytorch.*`. Committed signatures:

```python
Trainer(
    accelerator: str = "auto",         # "auto"|"cpu"|"gpu"|"clang"|"metal"|"cuda"|"llvm"
    devices: int | str | list[int] = 1,
    max_epochs: int | None = None,
    max_steps: int = -1,
    limit_train_batches: int | float = 1.0,
    limit_val_batches:   int | float = 1.0,
    val_check_interval:  int | float = 1.0,
    gradient_clip_val:   float | None = None,
    gradient_clip_algorithm: str = "norm",
    accumulate_grad_batches: int = 1,
    precision: str = "32-true",        # "32-true"|"16-mixed"|"bf16-mixed"
    callbacks: list[Callback] | None = None,
    logger: Logger | list[Logger] | bool = True,
    default_root_dir: str | None = None,
    enable_checkpointing: bool = True,
    enable_progress_bar: bool = True,
    deterministic: bool = False,
)
.fit(model, train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None)
.validate(model, dataloaders=None, datamodule=None, ckpt_path=None)
.test(model, dataloaders=None, datamodule=None, ckpt_path=None)

LightningModule:
  forward, training_step, validation_step, test_step
  configure_optimizers      # opt | (opts, scheds) | dict
  log(name, value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
  log_dict(...)
  save_hyperparameters(*args, ignore=...)
  parameters(), state_dict(), load_state_dict()

LightningDataModule(prepare_data, setup, train_dataloader, val_dataloader, test_dataloader)
```

`configure_optimizers` accepts the polymorphic PL return — including the dict form `{"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"|"epoch", "frequency": int, "monitor": "val_loss"}}`. Per-module LRs use param groups: `AdamW([{"params": m.encoder.parameters(), "lr": 1e-4}, {"params": m.head.parameters(), "lr": 1e-3}])`.

`self.log` accumulates per-step and reduces at epoch boundary. `LearningRateMonitor` reads from optimizer param groups. `ModelCheckpoint` writes safetensors (weights) plus a sidecar JSON (epoch, global_step, optimizer/scheduler state, hparams, callback states) and supports `monitor=`.

## Documented deltas from `lightning.pytorch`

These are intentional divergences. Parity tests acknowledge them via parametrization or device-skip.

### Devices

| PL value             | tinygrad-lightning behavior                                |
| -------------------- | ---------------------------------------------------------- |
| `"auto"`             | first available of `METAL`, `CUDA`, `CLANG`               |
| `"cpu"`              | force `CLANG` (or `LLVM` if available)                     |
| `"gpu"`              | first GPU-class device (`METAL`, `CUDA`, `AMD`)            |
| `"metal"`/`"cuda"`/`"clang"`/`"llvm"` | passthrough                                  |
| `devices > 1`        | `NotImplementedError("DDP not supported")`                 |

### Precision / AMP

PL's `"16-mixed"` uses `torch.autocast` + `GradScaler`. Tinygrad has no autocast. Our `PrecisionPlugin`:
- casts model parameters to compute dtype on forward, keeps fp32 master weights;
- multiplies loss by `loss_scale`, divides grads by `loss_scale` before optimizer step;
- detects non-finite grads, skips step + halves scale (DynamicLossScaler);
- `bf16-mixed` skips loss scaling entirely.

### Seeding

`seed_everything(seed)` sets `random`, `numpy.random`, `os.environ["PYTHONHASHSEED"]`, `Tensor.manual_seed`. Bit-exact reproducibility is guaranteed only same-device with `deterministic=True`; not across devices, not across backends.

### Checkpoint format

PL uses `torch.save` (pickle). We use safetensors + sidecar JSON:

```
epoch=02-step=000400.safetensors    # model.* + optimizer_<i>_<j>.* tensors
epoch=02-step=000400.meta.json      # epoch, global_step, scheduler_states, hparams, version
```

The safetensors blob carries both model weights (prefixed `model.`) and optimizer state tensors (prefixed `optimizer_<opt_idx>_<sub_idx>.`, where `sub_idx` indexes the param-group sub-optimizer). This covers Adam's `m`/`v`/`b1_t`/`b2_t` and SGD-with-momentum's `b` buffers — resume reproduces the contiguous training trajectory within `rtol=1e-3`. PL `.ckpt` files are not loadable.

### Optimizers / schedulers

Built on `tinygrad.nn.optim` (SGD, Adam, AdamW, LAMB). A thin `LightningOptimizer` wrapper accepts the PL-shaped `[{"params": ..., "lr": ...}]` and stores per-group LRs as scalars schedulers can mutate.

## Test strategy

Three folders, each with a different bar.

### `tests/parity/` — behavioral parity (B-tier)

One file per feature. Each test runs the same synthetic task (e.g. fitting a 2-layer MLP to a 64-sample regression set) twice — once via `lightning.pytorch`, once via `tinygrad_lightning` — asserting observable behavior matches. A pytest fixture `parity_backend` parametrizes each test with `["torch", "tinygrad"]`. Tests that can't sensibly compare across frameworks (e.g. exact MLflow run IDs) compare *within* each backend's own pre/post state.

Files and the assertions they make:

| File | Assertion |
|---|---|
| `test_basic_training.py` | `final_loss < 0.5 * initial_loss` after 1 epoch |
| `test_checkpointing.py` | resume after epoch 1 → finish epoch 2 → `global_step` and post-resume loss match a no-resume run within `1e-3` (per-backend, not cross-backend) |
| `test_lr_schedulers.py` | LR follows the scheduled curve at step boundaries |
| `test_param_group_lrs.py` | `head` weights change ≫ `encoder` weights when given different LRs |
| `test_mlflow_logger.py` | metric keys, step counts, tag set in the temp-file MLflow store match between backends |
| `test_data_loading.py` | `prepare_data → setup → train_dataloader` called in order, exactly once per stage |
| `test_devices.py` | `cpu` always runs; `gpu` skipped unless tinygrad picks a GPU device |
| `test_amp.py` | `16-mixed` completes 5 steps with no NaN losses; step counter advances |

### `tests/numeric/` — seed-pinned regressions (A-tier, narrow)

3–5 tests. Each: seed=42, fixed model, fixed data, run N steps, assert `loss < THRESHOLD` and a checksum of model weights matches a stored value. Guards against silent backend drift. Not run against PL.

### `tests/unit/` — standalone tinygrad-lightning unit tests

Cover wiring without a clean PL analogue: `PrecisionPlugin`, accelerator name resolution, safetensors checkpoint roundtrip, `LightningOptimizer` wrapper, scheduler interval handling.

### Test deps

- `lightning>=2.0`, `pytest`, `mlflow`, `tensorboardX` go in `requirements-test.txt`.
- `mlflow` is **also** a runtime optional extra: `extras_require={"mlflow": ["mlflow>=2.0"]}`. `MLFlowLogger` lazy-imports at construction with a clear error.
- `conftest.py` skips `tests/parity/` if `lightning` not importable, with a clear marker.

## Phasing

Six phases, each independently shippable. Each phase ends with green tests for *its* feature. Later phases don't block on later phases.

### Phase 0 — Foundations
- Bump `requirements.txt` to `tinygrad>=0.12.0`.
- Add `requirements-test.txt`.
- `setup.py`: setuptools-only, `python_requires=">=3.11"`, `extras_require`.
- `tests/conftest.py` with `parity_backend` fixture and PL-skip marker.
- Smoke test for public exports.
- Skeleton test files for phases 1–5, gated with `pytest.mark.skip("phase N")`.

### Phase 1 — Core training loop
- New `LightningModule` (PL hook shapes, `self.log` accumulator, `save_hyperparameters`).
- New `Trainer` with `fit`, `accelerator`/`devices` resolution, `max_epochs`/`max_steps`, `limit_*_batches`, `gradient_clip_val`, `accumulate_grad_batches`.
- `Callback` base + `TQDMProgressBar`.
- Activate `test_basic_training`, `test_devices` (cpu only), `test_loss_curves`.

### Phase 2 — Optimizers, schedulers, param groups
- `LightningOptimizer` wrapper, param-group plumbing, schedulers.
- `LearningRateMonitor`.
- Activate `test_lr_schedulers`, `test_param_group_lrs`.

### Phase 3 — Checkpointing
- `ModelCheckpoint` (top-K, `monitor=`, safetensors + meta).
- `Trainer.fit(..., ckpt_path=)` resume.
- Activate `test_checkpointing`.

### Phase 4 — Loggers + DataModule + EarlyStopping
- `Logger` ABC, `CSVLogger`, `TensorBoardLogger`, `MLFlowLogger` (lazy).
- `LightningDataModule` lifecycle.
- `EarlyStopping`.
- Activate `test_mlflow_logger`, `test_data_loading`.

### Phase 5 — AMP
- `PrecisionPlugin` + DynamicLossScaler.
- Activate `test_amp`.
