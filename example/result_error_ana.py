import os

from yolo_data_manager import YoloManager


def yolo_error_ana(input_dir, pred_dir, pred_name, workers=8):
    ana_dir = os.path.join(input_dir, "ana", pred_name)
    os.makedirs(ana_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="auto", init_check=False)

    # split
    mgr.eval_error_analysis(
        pred_root=pred_dir,
        out=ana_dir, 
        match_iou=0.5, 
        low_iou=0.1,
        conf_thres=0.01,
        duplicate_iou=0.9,
        review=True,
        crop_padding=12,
        review_workers=workers,
        review_progress=True,
        review_progress_leave=False,
        copy_pred_txt=True,
        )

if __name__ == "__main__":
    pass
    t_dir = r"/localnvme/data/bdd_hmt/sua_t"
    t_dir_pred = r"/localnvme/project/ultralytics/runs/detect/val-4/labels"
    rgb_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge"
    rgb_dir_pred = r"/localnvme/project/ultralytics/runs/detect/val-18/labels"
    # yolo_error_ana(t_dir, t_dir_pred, "val-4")
    yolo_error_ana(rgb_dir, rgb_dir_pred, "val-18")