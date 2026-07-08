import os

from yolo_data_manager import YoloManager


def yolo_split(input_dir):
    sta_dir = os.path.join(input_dir, "stats")
    os.makedirs(sta_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="flat", init_check=os.path.join(sta_dir, "validation.json"))

    # 统计
    mgr.stats(
        out=os.path.join(sta_dir, "stats.json"),
        class_csv=os.path.join(sta_dir, "class_counts.csv"),
    )
    mgr.dataset_duplicates(out=os.path.join(sta_dir, "duplicates.csv"))
    mgr.dataset_bad_images(out=os.path.join(sta_dir, "bad_images.csv"))
    # split
    mgr.dataset_split(train=0.9, val=0.1, test=0.0, seed=233, absolute_paths=True)

if __name__ == "__main__":
    pass
    cube_dir = r"/localnvme/data/bdd_hmt/bp_cube"
    rgb_all_dir = r"/localnvme/data/bdd_hmt/sua_rgb"
    t_all_dir = r"/localnvme/data/bdd_hmt/sua_t"
    # rgb_all_rgbt_dir = r"/localnvme/data/bdd_hmt/sua_rgb_rgbt"
    # t_all_rgbt_dir = r"/localnvme/data/bdd_hmt/sua_t_rgbt"

    yolo_split(cube_dir)
    yolo_split(rgb_all_dir)
    yolo_split(t_all_dir)
    # yolo_split(rgb_all_rgbt_dir)
    # yolo_split(t_all_rgbt_dir)