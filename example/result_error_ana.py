import os

from yolo_data_manager import YoloManager


def yolo_error_ana(input_dir, pred_dir, pred_name):
    ana_dir = os.path.join(input_dir, "ana", pred_name)
    os.makedirs(ana_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="auto", init_check=False)

    # split
    mgr.eval_error_analysis(
        gt_root=os.path.join(input_dir, "labels"),
        val_source=os.path.join(input_dir, "val.txt"),
        class_file=os.path.join(input_dir, "class.txt"),
        pred_root=pred_dir,
        out=ana_dir, 
        match_iou=0.5, 
        low_iou=0.1,
        conf_thres=0.01,
        duplicate_iou=0.9,
        review=True,
        crop_padding=12)

if __name__ == "__main__":
    pass
    cube_dir = r"/localnvme/data/bdd_hmt/bp_cube_filter_p01"
    rgb_all_dir = r"/localnvme/data/bdd_hmt/sua_rgb_filter_p01"
    cube_dir_pred = r"/localnvme/project/ultralytics/runs/detect/val181/labels"
    rgb_all_dir_pred = r"/localnvme/project/ultralytics/runs/detect/val180/labels"

    yolo_error_ana(cube_dir, rgb_all_dir_pred, "val180")
    yolo_error_ana(rgb_all_dir, rgb_all_dir_pred, "val180")
