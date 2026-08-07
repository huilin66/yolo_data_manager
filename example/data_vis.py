from yolo_data_manager import YoloManager


def yolo_vis(input_dir, crop=True):

    mgr = YoloManager(input_dir, layout="auto", init_check=False, init_layout=False)
    mgr.vis_draw(workers=8, show_id=True)

    if crop:
        mgr.vis_crop(workers=8, progress=True)


if __name__ == "__main__":
    # cube_dir = r"/localnvme/data/bdd_hmt/bp_cube"
    # rgb_merge_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge"
    # t_all_dir = r"/localnvme/data/bdd_hmt/sua_t"
    # rgb_merge_f02_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge_filter_0.02_AND"
    # yolo_vis(cube_dir)
    # yolo_vis(rgb_merge_f02_dir, crop=True)
    # yolo_vis(t_all_dir, crop=True)

    # data_dir = r"\\158.132.186.40\isds\huilin\traffic_sign\defect\detection\data_seg_1_damaged-guardrails"
    data_dir = (
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_bp_cube.yaml"
    )
    yolo_vis(data_dir)
