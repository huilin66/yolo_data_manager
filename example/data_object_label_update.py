from yolo_data_manager import YoloManager


def yolo_update(
    input_dir,
    crops_dir,
    to,
):
    mgr = YoloManager(input_dir, layout="auto", init_check=False)

    mgr.ann_correct_from_crops(
        crops_dir=crops_dir,
        to=to,
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
    crops_map = {
        "/localnvme/data/bdd_hmt/sua_rgb/image_vis/crop_change/2_b_low": "Broken Low Risk",
        "/localnvme/data/bdd_hmt/sua_rgb/image_vis/crop_change/2_e_low": "Efflorescene Low Risk",
    }

    for k, v in crops_map.items():
        yolo_update(
            data_dir,
            crops_dir=k,
            to=v,
        )
