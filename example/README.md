# Dataset examples

Files directly in this directory are dataset-specific callers. Copy
`dataset_template.py` to a name such as `hmt_rgb.py`, set that dataset's paths,
and select the manager methods to run.

The caller is intentionally small and explicit:

```python
from yolo_data_manager import YoloManager

DATA_DIR = r"/path/to/my_dataset.yaml"

# Create the manager once when several operations use the same dataset.
manager = YoloManager(DATA_DIR, layout="auto", init_check=False)
manager.stats(stats_list=["all"], only_val=False)
manager.vis_draw(only_val=False)
manager.vis_crop(only_val=False)
```

There is no generic dataset runner or wrapper-function layer. Each dataset
file owns its paths and operation parameters.
