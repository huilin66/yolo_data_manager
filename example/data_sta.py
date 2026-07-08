import os

from yolo_data_manager import YoloManager


def yolo_split(input_dir):
    sta_dir = os.path.join(input_dir, "stats", "labels_sta")
    os.makedirs(sta_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="flat", init_check=False)

    # split
    mgr.stats(plots_dir=sta_dir, stats_list=["all"])

if __name__ == "__main__":
    pass
    cube_dir = r"/localnvme/data/bdd_hmt/bp_cube"
    rgb_all_dir = r"/localnvme/data/bdd_hmt/sua_rgb"
    t_all_dir = r"/localnvme/data/bdd_hmt/sua_t"

    yolo_split(cube_dir)
    yolo_split(rgb_all_dir)
    yolo_split(t_all_dir)