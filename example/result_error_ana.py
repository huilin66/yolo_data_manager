import os

from yolo_data_manager import YoloManager


def yolo_error_ana(
    input_dir,
    pred_dir,
    pred_name,
    abs_path=False,
    only_val=True,
    workers=8,
    conf_thres=0.001,
    class_=None,
    exclude_class_=None,
    min_pixels=None,
    class_rules=None,
    **kwargs,
):
    if not abs_path:
        pred_dir = os.path.join(pred_dir, pred_name, "labels")

    mgr = YoloManager(input_dir, layout="auto", init_check=False)

    ana_dir = os.path.join(mgr.root, "result_ana", pred_name)
    mgr.eval_error_analysis(
        pred_root=pred_dir,
        out=ana_dir,
        conf_thres=conf_thres,
        crop_padding=12,
        review_workers=workers,
        only_val=only_val,
        class_=class_,
        exclude_class_=exclude_class_,
        min_pixels=min_pixels,
        class_rules=class_rules,
        workers=workers,
        **kwargs,
    )


if __name__ == "__main__":
    data_dir = (
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_bp_cube.yaml"
    )
    pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect"

    # yolo_error_ana(data_dir, pred_dir, "predict-6", only_val=False)
    yolo_error_ana(data_dir, pred_dir, "val-159", only_val=True)
