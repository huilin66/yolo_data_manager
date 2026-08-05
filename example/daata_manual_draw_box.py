from yolo_data_manager import YoloManager


def yolo_draw(input_dir):

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)
    mgr.vis_manual_box("DJI_20260211162423_0089.png")


if __name__ == "__main__":
    data_dir = (
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_rgb.yaml"
    )
    yolo_draw(data_dir)
