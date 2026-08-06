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
    min_pixels=None,
    conf_thres=0.001,
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
        min_pixels=min_pixels,
        conf_thres=conf_thres,
        out=os.path.join(ana_dir, "metrics.json"),
        csv=os.path.join(ana_dir, "metrics.csv"),
        print_table=True,
        show_original=True,
        only_val=True,
        workers=workers,
    )


if __name__ == "__main__":
    # data_dir = r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
    # pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect"
    # merge_class_map = {
    #     "Hollow": [
    #         "Hollow Low Risk",
    #         "Hollow High Risk",
    #     ],
    #     "Temperature": [
    #         "Temperature Medium Risk",
    #         "Temperature High Risk",
    #     ],
    # }
    # yolo_metric(
    #     data_dir,
    #     pred_dir,
    #     "val-52",
    #     merge_class_map=merge_class_map,
    # )

    data_dir = (
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_rgb.yaml"
    )
    data_update_dir = r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_rgb_update.yaml"
    pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect"
    merge_class_map = (
        {
            "Broken high": [
                "Broken High Risk",
            ],
            "Delamination": [
                # "Broken High Risk",
                "Delaminated Tile Low Risk",
                "Delaminate Tile High Risk",
                "Cracked Tile",
            ],
            "Efforescene": [
                "Efforescene Low Gray",
                # "Efflorescene Low Risk",
                "Efflorescene High Risk",
                # "Broken Low Risk",
            ],
            "Broken": [
                # "Broken Low Risk",
                "Efflorescene Low Risk",
            ],
        },
    )
    exclude_class_ = [
        "Broken Low Risk",
        # "Delaminated Tile Low Risk"
    ]
    # yolo_metric(
    #     data_dir,
    #     pred_dir,
    #     "val-53",
    #     merge_class_map=merge_class_map,
    # )
    yolo_metric(
        data_dir,
        pred_dir,
        "val-158",
        merge_class_map=merge_class_map,
        exclude_class_=exclude_class_,
        min_pixels=20,
        # conf_thres=0.10,
    )
    # yolo_metric(
    #     data_update_dir,
    #     pred_dir,
    #     "val-157",
    #     merge_class_map=merge_class_map,
    #     # exclude_class_=exclude_class_,
    #     min_pixels=20,
    #     conf_thres=0.20,
    # )
