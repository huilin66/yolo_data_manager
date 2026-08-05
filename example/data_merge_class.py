from yolo_data_manager import YoloManager


def yolo_merge_class(input_dir, output_dir, merge_dict):
    mgr = YoloManager(input_dir, layout="flat", init_check=False)
    mgr.ann_merge_class(merge_dict, out=output_dir)


if __name__ == "__main__":
    data_dir = r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
    data_update_dir = r"/localnvme/data/bdd_hmt/hmt_t_update_v1"
    merge_class_map = {
        "Hollow": [
            "Hollow Low Risk",
            "Hollow High Risk",
        ],
        "Temperature": [
            "Temperature Medium Risk",
            "Temperature High Risk",
        ],
    }
    yolo_merge_class(data_dir, data_update_dir, merge_class_map)
