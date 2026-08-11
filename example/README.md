# Dataset examples

`example/functions/` contains the reusable function layer.  Files directly in
this directory are dataset-specific callers: copy `dataset_template.py` to a
name such as `hmt_rgb.py`, set that dataset's paths, and select the functions
to run.

The caller is intentionally small and explicit:

```python
from yolo_data_manager import YoloManager
from example.functions import yolo_sta, yolo_vis

DATA_DIR = r"/path/to/my_dataset.yaml"

# Create the manager once when several operations use the same dataset.
manager = YoloManager(DATA_DIR, layout="auto", init_check=False)
yolo_sta(manager, stats_list=["all"], only_val=False)
yolo_vis(manager, crop=True, only_val=False)
```

The functions also continue to accept a dataset path.  Passing an existing
`YoloManager` reuses it and skips a second initialization.

There is no generic dataset runner and no path-editing task script.  Each
dataset file owns its paths and operation parameters.
