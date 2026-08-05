from yolo_data_manager import YoloManager


def yolo_update(
    input_dir,
    crops_dir,
    to,
):
    mgr = YoloManager(input_dir, layout="auto", init_check=False)

    mgr.ann_correct_from_error_crops(
        crops_dir=crops_dir,
        to=to,
    )


if __name__ == "__main__":
    data_dir = (
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_rgb.yaml"
    )
    crops_map = {
        "/localnvme/data/bdd_hmt/sua_rgb/result_ana/crop_change/broken_low_2_eff_low": "Efflorescene Low Risk",
        "/localnvme/data/bdd_hmt/sua_rgb/result_ana/crop_change/del_low_2_high": "Delaminate Tile High Risk",
        "/localnvme/data/bdd_hmt/sua_rgb/result_ana/crop_change/eff_low_2_broken_low": "Broken Low Risk",
        "/localnvme/data/bdd_hmt/sua_rgb/result_ana/crop_change/eff_low_2_del_low": "Delaminated Tile Low Risk",
        "/localnvme/data/bdd_hmt/sua_rgb/result_ana/crop_change/eff_low_2_gray": "Efforescene Low Gray",
    }

    for k, v in crops_map.items():
        yolo_update(
            data_dir,
            crops_dir=k,
            to=v,
        )
