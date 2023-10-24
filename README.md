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

#### The distribution of the size of the bounding boxes.
In the dataset, some smallest bounding boxes has area too small. These make it hard to detect and visualization. 
We better remove these bounding bonxes.

### Train-test split
- Split the training and validation. Need to ensure that there is no overlapping between the two sets.


## Proposed solutions:
### Solution 1: No-finetuning, only post-processing
### Solution 2: Fine-tunning to detect largest and smallest objects, then post-processing
### Solution 3: Fine-tuning to detect objects and non-objects, then post-processing
### Pre-processing boxes before NMS vs Post-processing boxes after NMS

## Metrics and results