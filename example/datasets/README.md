# Dataset runners

The files in `example/functions/` contain reusable operations.  The runner in
this directory accepts dataset paths at runtime, so a dataset can live
anywhere and the same command can be reused without editing source code.

Examples:

```powershell
# Analyze every image in one dataset (the default task is stats).
python -m example.datasets.run_dataset --data-dir E:\datasets\hmt_rgb --task stats

# Analyze several unrelated datasets in one invocation.
python -m example.datasets.run_dataset `
  --data-dir E:\datasets\hmt_rgb `
  --data-dir E:\datasets\hmt_t `
  --task stats

# Restrict an operation explicitly to validation data.
python -m example.datasets.run_dataset `
  --data-dir E:\datasets\hmt_rgb.yaml `
  --task vis --only-val --crop

# Ultralytics prediction run: --pred-dir is the runs/detect parent.
python -m example.datasets.run_dataset `
  --data-dir E:\datasets\hmt_rgb.yaml `
  --task metric `
  --pred-dir E:\models\runs\detect `
  --pred-name val-52 `
  --only-val --show-original --class Temperature
```

`--data-dir` accepts a dataset root or a dataset YAML path.  The same
function layer also covers visualization, filtering, splitting, class merge,
manual boxes, evaluation, and annotation correction; see `--help` for the
task-specific options.

