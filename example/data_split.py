import os

from yolo_data_manager import YoloManager


def yolo_split(input_dir, train=0.9, val=0.1, test=0.0, seed=233, absolute_paths=True):
    sta_dir = os.path.join(input_dir, "stats")
    os.makedirs(sta_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="flat", init_check=False)

    # split
    mgr.dataset_split(train=train, val=val, test=test, seed=seed, absolute_paths=absolute_paths)

if __name__ == "__main__":
    pass
    cube_dir = r"/localnvme/data/bdd_hmt/bp_cube"
    rgb_merge_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge"
    rgb_merge_f02_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge_filter_0.02_AND"
    rgb_merge_f02_f03_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge_filter_0.02_AND_filter_0.0_OR"
    t_dir = r"/localnvme/data/bdd_hmt/sua_t"
    rgb_merge_v2_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge_v2"

    # yolo_split(rgb_merge_dir)
    yolo_split(rgb_merge_v2_dir)
    # yolo_split(t_dir)

