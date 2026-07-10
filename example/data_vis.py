import os

from yolo_data_manager import YoloManager


def yolo_vis(input_dir, crop=False):
    vis_dir = os.path.join(input_dir, "image_vis")
    full_dir = os.path.join(vis_dir, "full")
    os.makedirs(vis_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)

    mgr.vis_draw(out=full_dir, workers=8, show_id=True)

    if crop:
        crop_dir = os.path.join(vis_dir, "crop")
        os.makedirs(crop_dir, exist_ok=True)
        mgr.vis_crop(out=crop_dir, workers=8, progress=True)


if __name__ == "__main__":
    pass
    cube_dir = r"/localnvme/data/bdd_hmt/bp_cube"
    rgb_merge_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge"
    t_all_dir = r"/localnvme/data/bdd_hmt/sua_t"
    rgb_merge_f02_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge_filter_0.02_AND"
    # yolo_vis(cube_dir)
    yolo_vis(rgb_merge_f02_dir, crop=True)
    # yolo_vis(t_all_dir, crop=True)