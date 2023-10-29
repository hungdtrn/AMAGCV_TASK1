import os
import shutil
import json
import random
import argparse
from collections import defaultdict
import numpy as np
import cv2
import pandas as pd
import pickle
from preprocess_utils import preprocess_boxes


def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]
        
def save(data_idx, boxPerImage, name, image_info, image_dir, data_dir, label_dir, single_class=False):
    fnames = []
    for i in data_idx:
        basename = f"{i:012d}"
        fnames.append(os.path.join(image_dir, f"{basename}.jpg"))
        
        w, h = image_info[i]["size"]
        boxes = boxPerImage[i]
        txtLabel = []
        for i, b in enumerate(boxes):
            x, y, bw, bh = b["bbox"]
            # cv2.rectangle(img, (int(x), int(y)), (int(x+bw), int(y+bh)), (0, 255, 0), 2)
            x, y, bw, bh = coco_to_yolo(x, y, bw, bh, w, h)
            
            if not single_class:
                ln = " ".join([str(i)] + [str(i) for i in [x, y, bw, bh]])
            else:
                ln = " ".join([str(0)] + [str(i) for i in [x, y, bw, bh]])
            txtLabel.append(ln)
        
        with open(os.path.join(label_dir, f"{basename}.txt"), "w") as f:
            f.write("\n".join(txtLabel))
            
    with open(os.path.join(data_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(fnames))

def divide_into_bins(keys, data, n_bins):
    labels = pd.qcut(data, n_bins)
    out = defaultdict(list)
    for i in range(len(labels)):
        out[labels[i]].append((keys[i], data[i]))
        
    return out

def computeArea(box):
    return box[-1] * box[-2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="dataset/raw/images")
    parser.add_argument("--cached_image_info", type=str, default="dataset/raw/image_info.p")
    parser.add_argument("--image_size", type=float, default=640)
    parser.add_argument("--raw_data_dir", type=str, default="dataset/raw")
    parser.add_argument("--output_dir")
    parser.add_argument("--min_smallest", type=float)
    parser.add_argument("--max_smallest", type=float)
    parser.add_argument("--min_largest", type=float)
    parser.add_argument("--max_largest", type=float)
    parser.add_argument("--min_size", type=float)

    args = parser.parse_args()
    
    twoBox_twoClass_dir = os.path.join(args.output_dir, "2box2class")
    twoBox_oneClass_dir = os.path.join(args.output_dir, "2box1class")
    
    all_box_dir = os.path.join(args.output_dir, "allboxallclass")
    all_box_oneClass_dir = os.path.join(args.output_dir, "allbox1class")
    
    
    annotation_dir_path = os.path.join(args.raw_data_dir, "annotations")
    with open(os.path.join(annotation_dir_path, "instances_val2017.json"), "r") as f:
        annotations = json.load(f)
                
    # Get image info
    print('------ Getting image info ----------')
    image_info = None
    if os.path.exists(args.cached_image_info):
        with open(args.cached_image_info, "rb") as f:
            image_info = pickle.load(f)
        
        if image_info["__image_size"] != args.image_size:
            image_info = None
            
    if image_info is None:
        image_info = {"__image_size": args.image_size}
        for i in [x["image_id"] for x in annotations["annotations"]]:
            basename = f"{i:012d}"        
            img = cv2.imread(os.path.join(args.image_dir, f"{basename}.jpg"))
            h, w = img.shape[:2]
            image_info[i] = {
                "size": [w, h],
                "ratio": args.image_size / max(h, w),
            }
            
        with open(args.cached_image_info, "wb") as f:
            pickle.dump(image_info, f)
                
                
    print("------ Preparing the labels ----------")
    minMaxPerImage = preprocess_boxes(annotations,
                                      image_info,
                                   min_smallest=args.min_smallest,
                                   max_smallest=args.max_smallest,
                                   min_largest=args.min_largest,
                                   max_largest=args.max_largest,
                                   min_size=args.min_size,
                                   num_objects=2)
    allPerImage = preprocess_boxes(annotations,
                                   image_info,
                                   min_smallest=args.min_smallest,
                                   max_smallest=args.max_smallest,
                                   min_largest=args.min_largest,
                                   max_largest=args.max_largest,
                                   min_size=args.min_size,
                                   num_objects=-1)
    
    
    # Train test split    
    print('------ Train/validation split ----------')
    all_videos = list(minMaxPerImage.keys())
    
    all_smallest_boxes = np.array([computeArea(k[0]["bbox"]) for k in minMaxPerImage.values()])
    all_largest_boxes = np.array([computeArea(k[1]["bbox"]) for k in minMaxPerImage.values()])
    ratio = all_smallest_boxes / all_largest_boxes
    groups = divide_into_bins(all_videos, ratio, 20)
    val = []
    for k in groups:
        videos_per_group = [i[0] for i in groups[k]]
        group_ratio = [ratio[all_videos.index(i)] for i in videos_per_group]
        print(f"Group {k}: {len(videos_per_group)}, min: {min(group_ratio)}, max: {max(group_ratio)}, mean: {np.mean(group_ratio)}, std: {np.std(group_ratio)}")
        np.random.shuffle(videos_per_group)
        n_val = int(0.1 * len(videos_per_group))
        val.extend(videos_per_group[:n_val])
    
    all_videos, val = set(all_videos), set(val)
    train = all_videos.difference(val)
    
    print(f"Total: {len(all_videos)}, train: {len(train)}, val: {len(val)}")

    
    # Save to files
    print("------ Saving to files ----------")
    out_dir_list = [twoBox_twoClass_dir, twoBox_oneClass_dir, all_box_dir, all_box_oneClass_dir]
    data_list = [minMaxPerImage, minMaxPerImage, allPerImage, allPerImage]
    single_class_list = [False, True, False, True]
    
    for out_dir, data, single_flag in zip(out_dir_list, data_list, single_class_list):
        os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
        save(train, data, "train", image_info, args.image_dir, out_dir, os.path.join(out_dir, "labels"), single_class=single_flag)
        save(val, data, "val", image_info, args.image_dir, out_dir, os.path.join(out_dir, "labels"), single_class=single_flag)