Requirement:
1. Download the data & annotations. 
2. Prepocessing the data  
3. Change the architecture to only detect the biggest and the smallest objects.

## Preprequisite
Insteall the environment
```
conda create -n amagcv python=3.9
pip install -r requirements.txt
```

## Processing the data
The number of all data is 5000.

### Select the biggest and the smallest objects 
- Get all bounding boxes for each images
- Get the largest and the smallest boxes, remove all other boxes. Assign the new labels to the two boxes
- Convert the box format to the yolo format and normalize
- IMPORTANT: Remove images that only have 1 bounding box because they will introduce noise.

### Train-test split
- Split the training and validation. Need to ensure that there is no overlapping between the two sets.


## Possible solutions:
### Solution 1: Finetuning the whole model
- This can serve as the baseline
- In the default model of YOLOV7, the anchors are automatically learned from the training data, so we hope that the model can learn a good set of achors for detecting the smallest and the largest objects.
- 

### Solution 2: Only Finetune the head
- Since the training data is limited and the training time is also limted (20 epochs), it's not a good idea to fine-tuning the whole model. 
- This training data is also COCO. The visual features are not much difference from the pre-training data. => We can re-use and freezing the backbone, fine-tuning only the head.
- Experiment show that it is slightly better than finetuning the whole model, which is expected.
- However, from the visualization, below, it seems that fine-tuning the head alone does not ensure that the model only detect 2 objects

![alt text](https://github.com/hungdtrn/AMAGCV_TASK1/blob/main/public/finetune_head/horses.jpg?raw=true)
![alt text](https://github.com/hungdtrn/AMAGCV_TASK1/blob/main/public/finetune_head/horses.jpg?raw=true)
