import os

from yolo_data_manager import YoloManager


def yolo_select_val(input_dir, copy_images=True):
    mgr = YoloManager(input_dir, layout="flat", init_check=False)

    mgr.dataset_select(
        file=os.path.join(mgr.root, "val.txt"),
        out=os.path.join(mgr.root, "subset", "val"),
        copy_images=copy_images,
    )


if __name__ == "__main__":
    pass
    data_dir = r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_traffic_sign/tf_defect_3.yaml"

    yolo_select_val(data_dir, copy_images=True)
