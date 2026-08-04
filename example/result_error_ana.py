import os

from yolo_data_manager import YoloManager


def yolo_error_ana(input_dir, pred_dir, pred_name, abs_path=False, workers=8):
    if not abs_path:
        pred_dir = os.path.join(pred_dir, pred_name, "labels")

    mgr = YoloManager(input_dir, layout="auto", init_check=False)

    ana_dir = os.path.join(mgr.root, "result_ana", pred_name)
    mgr.eval_error_analysis(
        pred_root=pred_dir,
        out=ana_dir,
        conf_thres=0.01,
        # match_iou=0.5,
        # low_iou=0.1,
        # duplicate_iou=0.9,
        review=True,
        crop_padding=12,
        review_workers=workers,
        review_progress=True,
        review_progress_leave=False,
        copy_pred_txt=True,
        only_val=True,
        workers=workers,
    )


if __name__ == "__main__":
    data_dir = r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
    pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect"

    yolo_error_ana(data_dir, pred_dir, "predict")
