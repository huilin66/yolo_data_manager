import os

from yolo_data_manager import YoloManager


def yolo_split(input_dir):
    out_dir = input_dir+'_filter_p01'
    os.makedirs(out_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)

    mgr.dataset_filter(
        out=out_dir,
        min_width=0.01,
        min_height=0.01,
    )


if __name__ == "__main__":
    pass
    cube_dir = r"/localnvme/data/bdd_hmt/bp_cube"
    rgb_all_dir = r"/localnvme/data/bdd_hmt/sua_rgb"
    t_all_dir = r"/localnvme/data/bdd_hmt/sua_t"

    yolo_split(cube_dir)
    yolo_split(rgb_all_dir)
    yolo_split(t_all_dir)