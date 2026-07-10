import os

from yolo_data_manager import YoloManager


def yolo_merge_class(input_dir, output_dir, merge_dict):
    mgr = YoloManager(input_dir, layout="flat", init_check=False)
    mgr.ann_merge_class(merge_dict, out=output_dir)

if __name__ == "__main__":
    pass
    rgb_dir = r"/localnvme/data/bdd_hmt/sua_rgb"
    rgb_merge_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge"
    rgb_merge_v2_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge_v2"
    rgb_merge = {
        "Broken High Risk":["Broken High Risk", "Cracked Tile"],
        "Corrosion": ["Corrosion", "Spalling"],
        "Efflorescene Low Risk": ["Efflorescene Low Risk", "Broken Low Risk"], 
        }
    rgb_merge_v2 = {
        "Delaminated Tile":["Delaminated Tile Low Risk", "Delaminate Tile High Risk"],
        }
    # yolo_merge_class(rgb_dir, rgb_merge_dir, rgb_merge)
    yolo_merge_class(rgb_merge_dir, rgb_merge_v2_dir, rgb_merge_v2)
