# Dataset examples

`example/functions/` contains the reusable function layer.  Files directly in
this directory are dataset-specific callers: copy `dataset_template.py` to a
name such as `hmt_rgb.py`, set that dataset's paths, and select the functions
to run.

The caller is intentionally small and explicit:

```python
from example.functions import yolo_sta, yolo_vis

DATA_DIR = r"/path/to/my_dataset.yaml"

yolo_sta(DATA_DIR, stats_list=["all"], only_val=False)
yolo_vis(DATA_DIR, crop=True, only_val=False)
```

There is no generic dataset runner and no path-editing task script.  Each
dataset file owns its paths and operation parameters.

