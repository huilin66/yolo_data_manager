import os

from yolo_data_manager import YoloManager


def yolo_sta(input_dir):
    sta_dir = os.path.join(input_dir, "stats", "labels_sta")
    os.makedirs(sta_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="flat", init_check=False)

    # split
    mgr.stats(plots_dir=sta_dir, stats_list=["all"])

if __name__ == "__main__":
    pass
    t_all_dir = r"/localnvme/data/bdd_hmt/sua_t"

    yolo_sta(t_all_dir)