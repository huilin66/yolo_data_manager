from yolo_data_manager import YoloManager


def yolo_draw(input_dir, image_name):

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)
    mgr.vis_manual_box(
        image_name,
        out="demo/manual_box_result.json",
    )


if __name__ == "__main__":
    data_dir = (
        r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_rgb.yaml"
    )
    image_name = "DJI_20260211163524_0337.png"
    yolo_draw(data_dir, image_name)
