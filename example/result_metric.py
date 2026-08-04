import os

from yolo_data_manager import YoloManager


def yolo_metric(
    input_dir,
    pred_dir,
    pred_name,
    abs_path=False,
    workers=8,
    class_=None,
    exclude_class_=None,
    merge_class_map=None,
):
    if not abs_path:
        pred_dir = os.path.join(pred_dir, pred_name, "labels")

    mgr = YoloManager(input_dir, layout="auto", init_check=False)
    ana_dir = os.path.join(mgr.root, "result_ana", pred_name)
    mgr.eval_metrics(
        pred_root=pred_dir,
        class_=class_,  # 可选：只评估指定类别；也可用 [0, 1]
        exclude_class_=exclude_class_,
        merge_class_map=merge_class_map,
        # min_pixels=8,  # 可选：过滤小目标
        conf_thres=0.001,  # 可选：置信度过滤
        out=os.path.join(ana_dir, "metrics.json"),
        csv=os.path.join(ana_dir, "metrics.csv"),
        print_table=True,
        show_original=True,
        only_val=True,
        workers=workers,
    )


if __name__ == "__main__":
    data_dir = r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
    pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect"

    yolo_metric(
        data_dir,
        pred_dir,
        "val-52",
        # class_=["Hollow High Risk Line"],
        exclude_class_=["Hollow High Risk Line"],
        merge_class_map={
            "Hollow": [
                "Hollow Low Risk",
                "Hollow High Risk",
            ],
            "Temperature": [
                "Temperature Medium Risk",
                "Temperature High Risk",
            ],
        },
    )
