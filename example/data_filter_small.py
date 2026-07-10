import os

from yolo_data_manager import YoloManager


def yolo_filter_small(input_dir, filter_ratio=0.01):
    out_dir = input_dir+'_filter_'+str(filter_ratio)
    os.makedirs(out_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)

    mgr.dataset_filter(
        out=out_dir,
        min_width=filter_ratio,
        min_height=filter_ratio,
    )


if __name__ == "__main__":
    pass
    # cube_dir = r"/localnvme/data/bdd_hmt/bp_cube"
    # rgb_all_dir = r"/localnvme/data/bdd_hmt/sua_rgb"
    t_all_dir = r"/localnvme/data/bdd_hmt/sua_t"

    # yolo_filter_small(cube_dir)
    # yolo_filter_small(rgb_all_dir)
    yolo_filter_small(t_all_dir, filter_ratio=0.05)