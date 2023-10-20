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

Install and setup wandb [here](https://docs.wandb.ai/quickstart)

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
### Solution 1: Finetuning the whole model - Baseline
- This can serve as the baseline
- In the default model of YOLOV7, the anchors are automatically learned from the training data, so we hope that the model can learn a good set of achors for detecting the smallest and the largest objects.
- 

### Solution 2: Only Finetune the head
- Due to the limited training data and a restricted training time of 20 epochs, it's not a good idea to fine-tuning the whole model. 
- The training dataset used here is COCO, and the visual features are not significantly different from the pre-training data. We can reuse and freeze the backbone while fine-tuning only the model's head.
- Experimental results demonstrate a slight improvement over fine-tuning the entire model, as expected.
- However, from the visualization, below, it seems that fine-tuning only the model's head does not guarantee that the model will exclusively detect two objects.

![alt text](https://github.com/hungdtrn/AMAGCV_TASK1/blob/main/public/finetune_head/horses.jpg?raw=true)
![alt text](https://github.com/hungdtrn/AMAGCV_TASK1/blob/main/public/finetune_head/image1.jpg?raw=true)

### Solution 3: Relativity-aware head
- The sizes of the smallest and largest boxes vary depending on the images.
- To determine whether a bounding box is the largest or smallest, we require a mechanism to consider all other bounding box candidates in the image.
- Attention mechanism appears to be a promising choice for this task.