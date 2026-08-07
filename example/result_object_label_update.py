from yolo_data_manager import YoloManager


def yolo_update(input_dir, crops_dir, to, pred_dir=None):
    mgr = YoloManager(input_dir, layout="auto", init_check=False)

    mgr.ann_correct_from_error_crops(
        crops_dir=crops_dir,
        to=to,
        pred_dir=pred_dir,
        # replace_gt_from_pred=True,
    )


if __name__ == "__main__":
    import os
    # data_dir = (
    #     r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_rgb.yaml"
    # )
    # # pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/predict-5"
    # pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/val-158"
    # # crop_change_dir = r"/localnvme/data/bdd_hmt/sua_rgb/result_ana/crop_change2"
    # crop_change_dir = r"/localnvme/data/bdd_hmt/sua_rgb/result_ana/crop_change3"
    # crops_map = {
    #     # os.path.join(crop_change_dir, "b_high_2_d_high"): "Delaminate Tile High Risk",
    #     # os.path.join(crop_change_dir, "b_low_2_e_low"): "Efflorescene Low Risk",
    #     # os.path.join(crop_change_dir, "d_high_2_b_low"): "Broken Low Risk",
    #     # os.path.join(crop_change_dir, "e_high_2_low"): "Efflorescene Low Risk",
    #     # os.path.join(crop_change_dir, "e_low_2_b_low"): "Broken Low Risk",
    #     # os.path.join(crop_change_dir, "e_low_2_gray"): "Efforescene Low Gray",
    #     # os.path.join(crop_change_dir, "e_low_2_high"): "Efflorescene High Risk",
    #     # os.path.join(crop_change_dir, "b_low_2_none"): None,
    #     # os.path.join(crop_change_dir, "d_low_2_none"): None,
    #     # os.path.join(crop_change_dir, "e_low_2_none"): None,
    #     os.path.join(crop_change_dir, "none_2_b_high"): "Broken High Risk",
    #     os.path.join(crop_change_dir, "none_2_b_low"): "Broken Low Risk",
    #     os.path.join(crop_change_dir, "none_2_c"): "Corrosion",
    #     os.path.join(crop_change_dir, "none_2_d_high"): "Delaminate Tile High Risk",
    #     os.path.join(crop_change_dir, "none_2_d_low"): "Delaminated Tile Low Risk",
    #     os.path.join(crop_change_dir, "none_2_e_gray"): "Efforescene Low Gray",
    #     os.path.join(crop_change_dir, "none_2_e_low"): "Efflorescene Low Risk",
    # }
    # crops_map = {
    #     # os.path.join(crop_change_dir, "2_b_high"): "Broken High Risk",
    #     # os.path.join(crop_change_dir, "2_b_low"): "Broken Low Risk",
    #     # os.path.join(crop_change_dir, "2_c"): "Corrosion",
    #     # os.path.join(crop_change_dir, "2_e_gray"): "Efforescene Low Gray",
    #     # os.path.join(crop_change_dir, "2_e_high"): "Efflorescene High Risk",
    #     # os.path.join(crop_change_dir, "2_e_low"): "Efflorescene Low Risk",
    #     # os.path.join(crop_change_dir, "gt_2_pred"): None,
    #     # os.path.join(crop_change_dir, "none_2_b_high"): "Broken High Risk",
    #     # os.path.join(crop_change_dir, "none_2_b_low"): "Broken Low Risk",
    #     # os.path.join(crop_change_dir, "none_2_d_high"): "Delaminate Tile High Risk",
    #     # os.path.join(crop_change_dir, "none_2_d_low"): "Delaminated Tile Low Risk",
    #     # os.path.join(crop_change_dir, "none_2_e_gray"): "Efforescene Low Gray",
    #     # os.path.join(crop_change_dir, "none_2_e_low"): "Efflorescene Low Risk",
    # }
    # crops_map = {
    #     # os.path.join(crop_change_dir, "2_b_low"): "Broken Low Risk",
    #     # os.path.join(crop_change_dir, "2_e_low"): "Efflorescene Low Risk",
    #     # os.path.join(crop_change_dir, "gt_2_pred"): None,
    #     os.path.join(crop_change_dir, "none_2_c"): "Corrosion",
    #     os.path.join(crop_change_dir, "none_2_d_low"): "Delaminated Tile Low Risk",
    #     os.path.join(crop_change_dir, "none_2_e_gray"): "Efforescene Low Gray",
    #     os.path.join(crop_change_dir, "none_2_e_low"): "Efflorescene Low Risk",
    # }
    # for k, v in crops_map.items():
    #     yolo_update(
    #         data_dir,
    #         crops_dir=k,
    #         to=v,
    #         pred_dir=pred_dir,
    #     )

    data_dir = (
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_bp_cube.yaml"
    )
    pred_dir = r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/predict-6"
    crop_change_dir = (
        r"/localnvme/data/bdd_hmt/bp_cube/result_ana/predict-6/crop_change"
    )
    crops_map = {
        os.path.join(crop_change_dir, "none_2_b"): "broken",
        os.path.join(crop_change_dir, "none_2_e"): "efflorescene",
        os.path.join(crop_change_dir, "none_2_p"): "peeling",
    }
    for k, v in crops_map.items():
        yolo_update(
            data_dir,
            crops_dir=k,
            to=v,
            pred_dir=pred_dir,
        )
