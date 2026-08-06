import os

from yolo_data_manager import YoloManager


def yolo_error_ana(
    input_dir,
    pred_dir,
    pred_name,
    abs_path=False,
    only_val=True,
    workers=8,
    class_=None,
    exclude_class_=None,
    min_width=None,
    min_height=None,
    min_area=None,
    min_size_logic="or",
    min_pixels=None,
):
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
        only_val=only_val,
        class_=class_,
        exclude_class_=exclude_class_,
        min_width=min_width,
        min_height=min_height,
        min_area=min_area,
        min_size_logic=min_size_logic,
        min_pixels=min_pixels,
        workers=workers,
    )


if __name__ == "__main__":
    data_dir = (
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_rgb.yaml"
    )
    pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect"

    yolo_error_ana(data_dir, pred_dir, "predict-3", only_val=False)
