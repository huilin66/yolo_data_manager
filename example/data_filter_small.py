import os

from yolo_data_manager import YoloManager


def yolo_filter_small(input_dir, filter_ratio=0.01, logic='or', class_rules=None):
    out_dir = input_dir+f'_filter_{filter_ratio}_{logic.upper()}'
    os.makedirs(out_dir, exist_ok=True)

    mgr = YoloManager(input_dir, layout="flat", init_check=False, init_layout=False)

    if class_rules is None:
        mgr.dataset_filter(
            out=out_dir,
            min_width=filter_ratio,
            min_height=filter_ratio,
            min_size_logic=logic,
        )
    else:
        mgr.dataset_filter(
            out=out_dir,
            class_rules=class_rules
        )

if __name__ == "__main__":
    pass
    # rgb_merge_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge"
    # yolo_filter_small(rgb_merge_dir, filter_ratio=0.02, logic='and')
    # rgb_merge_f02_dir = r"/localnvme/data/bdd_hmt/sua_rgb_merge_filter_0.02_AND"
    # yolo_filter_small(rgb_merge_f02_dir, filter_ratio=0.0, logic='or',
    #     class_rules={
    #         'Efflorescene Low Risk':{
    #                             "min_width": 0.03,
    #                             "min_height": 0.03,
    #                             "min_size_logic": "or",
    #                         },

    #     }
    # )