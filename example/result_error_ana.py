import os

from yolo_data_manager import YoloManager


def yolo_error_ana(input_dir, pred_dir, pred_name, abs_path=False, workers=8):
    if not abs_path:
        pred_dir = os.path.join(input_dir, pred_dir, 'labels')

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
    data_dir = r'/localnvme/project/ultralytics/ultralytics/cfg/datasets_traffic_sign/tf_defect_3.yaml'
    pred_dir = r"/localnvme/project/ultralytics/runs/detect"

    yolo_error_ana(data_dir, pred_dir, "val-23")