import os

from yolo_data_manager import YoloManager


def yolo_merge_class(input_dir, output_dir, merge_dict):
    mgr = YoloManager(input_dir, layout="flat", init_check=False)
    mgr.ann_merge_class(merge_dict, out=output_dir)

if __name__ == "__main__":
    pass
    rgb_f01_dir = r"/localnvme/data/bdd_hmt/sua_rgb_filter_p01"
    rgb_f01_merge_dir = r"/localnvme/data/bdd_hmt/sua_rgb_filter_p01_merge"
    t_f01_dir = r"/localnvme/data/bdd_hmt/sua_t_filter_p01"
    t_f01_merge_dir = r"/localnvme/data/bdd_hmt/sua_t_filter_p01_merge_v2"
    rgb_merge = {
        "Broken": ["Broken Low Risk", "Broken High Risk", "Cracked Tile"], 
        "Corrosion": ["Corrosion", "Spalling", "background"],
        "Delaminated": ["Delaminated Tile Low Risk", "Delaminate Tile High Risk"],
        "Efforescene": ["Efforescene Low Gray", "Efflorescene Low Risk", "Efflorescene High Risk"],
        }
    t_merge = {
        "Hollow": ["Hollow Low Risk", "Hollow High Risk", "Hollow High Risk Line"], 
        "Leakage": ["Leakage High Risk", "background"],
        }



    # yolo_merge_class(rgb_f01_dir, rgb_f01_merge_dir, rgb_merge)
    yolo_merge_class(t_f01_dir, t_f01_merge_dir, t_merge)