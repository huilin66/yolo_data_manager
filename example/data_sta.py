import os

from yolo_data_manager import YoloManager


def yolo_sta(input_dir):

    mgr = YoloManager(input_dir, layout="flat", init_check=False)
    sta_dir = os.path.join(mgr.root, "stats", "labels_sta")
    os.makedirs(sta_dir, exist_ok=True)
    # split
    mgr.stats(plots_dir=sta_dir, stats_list=["all"])


if __name__ == "__main__":
    # t_all_dir = r"\\158.132.186.40\isds\huilin\traffic_sign\defect\detection\data_seg_1_damaged-guardrails"
    data_dir = r"/localnvme/project/ultralytics/ultralytics/cfg/datasets_hmt/hmt_t.yaml"
    yolo_sta(data_dir)
