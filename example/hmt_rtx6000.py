import os

from yolo_data_manager import YoloManager


def yolo_split(input_dir):
    sta_dir = os.path.join(input_dir, "stats")
    os.makedirs(sta_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="auto")

    # 校验
    mgr.check(out=os.path.join(sta_dir, "validation.json"))

    # 统计
    mgr.stats(
        out=os.path.join(sta_dir, "stats.json"),
        class_csv=os.path.join(sta_dir, "class_counts.csv"),
    )


if __name__ == "__main__":
    rgb_all_dir = r"/lovalnvme/data/bdd_hmt/sua_dataset_rgb_all"
    rgb_all_rgbt_dir = r"/lovalnvme/data/bdd_hmt/sua_dataset_rgb_all_rgbt"
    t_all_dir = r"/lovalnvme/data/bdd_hmt/sua_dataset_t_all"
    t_all_rgbt_dir = r"/lovalnvme/data/bdd_hmt/sua_dataset_t_all_rgbt"

    yolo_split(r"E:\datasets\my_yolo")
