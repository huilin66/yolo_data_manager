import os

from yolo_data_manager import YoloManager


def yolo_metric(input_dir, pred_dir, pred_name, abs_path=False, workers=8):
    if not abs_path:
        pred_dir = os.path.join(pred_dir, pred_name, "labels")

    mgr = YoloManager(input_dir, layout="auto", init_check=False)
    ana_dir = os.path.join(mgr.root, pred_name)
    mgr.eval_metrics(
        pred_root=pred_dir,
        # class_=["fire", "smoke"],   # 可选：只评估指定类别；也可用 [0, 1]
        # min_pixels=8,               # 可选：过滤小目标
        conf_thres=0.1,  # 可选：置信度过滤
        out=os.path.join(ana_dir, "metrics.json"),
        csv=os.path.join(ana_dir, "metrics.csv"),
        print_table=True,
    )


if __name__ == "__main__":
    pass
    data_dir = r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_traffic_sign/tf_defect_3.yaml"
    pred_dir = r"/localnvme/project/ultralytics/runs/detect"

    yolo_metric(data_dir, pred_dir, "val-23")
