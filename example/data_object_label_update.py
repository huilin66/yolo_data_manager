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
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_bp_cube.yaml"
    )
    crops_map = {
        "/localnvme/data/bdd_hmt/bp_cube/image_vis/crop_change/2_p": "peeling",
    }

    for k, v in crops_map.items():
        yolo_update(
            data_dir,
            crops_dir=k,
            to=v,
        )
