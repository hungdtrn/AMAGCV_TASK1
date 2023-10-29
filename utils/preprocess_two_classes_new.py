import os
import shutil
import json
import random
import argparse
from collections import defaultdict
import numpy as np
import cv2
from preprocess_utils import preprocess_boxes
import pandas as pd

def divide_into_bins(keys, data, n_bins):
    labels = pd.qcut(data, n_bins)
    out = defaultdict(list)
    for i in range(len(labels)):
        out[labels[i]].append((keys[i], data[i]))
        
    return out
        

def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]
        
def save(data_idx, boxPerImage, name, image_dir, data_dir, label_dir):
    fnames = []
    for i in data_idx:
        basename = f"{i:012d}"
        fnames.append(os.path.join(image_dir, f"{basename}.jpg"))
        
        img = cv2.imread(os.path.join(image_dir, f"{basename}.jpg"))
        h, w = img.shape[:2]
        boxes = boxPerImage[i]
        txtLabel = []
        for i, b in enumerate(boxes):
            x, y, bw, bh = b["bbox"]
            # cv2.rectangle(img, (int(x), int(y)), (int(x+bw), int(y+bh)), (0, 255, 0), 2)
            x, y, bw, bh = coco_to_yolo(x, y, bw, bh, w, h)
            txtLabel.append(" ".join([str(i)] + [str(i) for i in [x, y, bw, bh]]))
        
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        with open(os.path.join(label_dir, f"{basename}.txt"), "w") as f:
            f.write("\n".join(txtLabel))
            
    with open(os.path.join(data_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(fnames))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="dataset/raw/images")
    parser.add_argument("--raw_data_dir", type=str, default="dataset/raw")
    parser.add_argument("--output_dir")
    parser.add_argument("--min_smallest", type=float)
    parser.add_argument("--max_smallest", type=float)
    parser.add_argument("--min_largest", type=float)
    parser.add_argument("--max_largest", type=float)

    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels/val"), exist_ok=True)

    annotation_dir_path = os.path.join(args.raw_data_dir, "annotations")
    with open(os.path.join(annotation_dir_path, "instances_val2017.json"), "r") as f:
        annotations = json.load(f)
                
    def computeArea(box):
        return box[-1] * box[-2]

    boxPerImage = preprocess_boxes(annotations,
                                   min_smallest=args.min_smallest,
                                   max_smallest=args.max_smallest,
                                   min_largest=args.min_largest,
                                   max_largest=args.max_largest)
    
    # Train test split    
    all_videos = list(boxPerImage.keys())
    all_smallest_boxes = all_smallest = np.array([computeArea(k[0]["bbox"]) for k in boxPerImage.values()])
    groups = divide_into_bins(all_videos, all_smallest_boxes, 10)
    val = []
    for k in groups:
        videos_per_group = [i[0] for i in groups[k]]
        np.random.shuffle(videos_per_group)
        n_val = int(0.1 * len(videos_per_group))
        val.extend(videos_per_group[:n_val])
        

    all_videos, val = set(all_videos), set(val)
    train = all_videos.difference(val)
        
    # np.random.shuffle(all_videos)
    # n_val = int(0.1 * len(all_videos))
    # val = all_videos[:n_val]
    # train = all_videos[n_val:]

    print(len(train), len(val))
        
    # Save to files
    save(train, boxPerImage, "train", args.image_dir, args.output_dir, os.path.join(args.output_dir, "labels"))
    save(val, boxPerImage, "val", args.image_dir, args.output_dir, os.path.join(args.output_dir, "labels"))       