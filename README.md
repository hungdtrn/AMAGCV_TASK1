# Report for Computer Vision Task: Detecting the largest and the smallest objects in Images / Videos using YOLO.
This report outlines my approaches to data preparation and fine-tuning the YOLO-V7 model for detecting the largest and smallest objects in images and videos.

## 1. Summary
### 1.1. About the data
- There is no clear distinction between smallest and largest bounding boxes. Sizes alone cannot determine this distiction; comparison between boxes in the same image is necessary
- Some objects are exceptionally small (as small as 2 pixels), making accurate labeling difficult, and these tiny boxes are predominantly noise in the data. Around 98% of the data has the smallest bounding box larger than 17 square pixels, indicating a need for a lower limit on the smallest bounding box area. This could aid learning and visualization.
- Instances where there is only one groundtruth object in the data have been treated as noise and removed from the dataset.
### 1.2. About the models
- No Finetuning, Post-processing Only: YOLO-v7 output bounding boxes were post-processed without finetuning. This method was slow and inefficient because it yields an excessive number of unnecessary bounding boxes. Consequently, NMS is applied to a larger set of box candidates than needed.
- Fine-tuning YOLO-V7 for single-class object detection using annotations for the largest and smallest objects, followed by post-processing to isolate and identify only the largest and smallest objects in the image. The fine-tuned model generates a smaller set of bounding boxes candidates,leading to quicker NMS. Two variants of this approach are considered: (1) Fine-tuning the entire model and (2) fine-tuning only the detection head. Option (2) is more cost-effective and accurate.
- Given that the task now solely involves detecting the largest and smallest bounding boxes in an image, these boxes can be selected from the bounding box candidates before applying NMS. This reduces the time spent on NMS but comes at the cost of reduced accuracy since the largest boxes may not necessarily be the ones with the most significant overlap with the object.

## 2. Preprequisite Installation
Install the environment
```
conda create -n amagcv python=3.9
conda activate amagcv
pip install -r requirements.txt
```
Install and setup wandb as instructed [here](https://docs.wandb.ai/quickstart)

Download the dataset and model and unzip to the desired folder
```
bash download.sh 
```
## 3. Project structures
Most of this projects were copied from the original git repo of YOLO-v7. For details of the structure of the project, please visit the original repo.
Here are the added folders/files:
- `download.sh`: The bash file used to download the required dataset and model
- `tools/data_analysis.ipynb`: Notebook for analyze the size of the bounding boxes 
- `utils/prepare_data.py`: The python file used to prepare the dataset that contains two labeled bounding boxes: largest and smalelst
- `utils/prepare_data_single_class.py`: The python file used to prepare the dataset that contains a single label showing whether the box is object or not (single-class)
- `data/custom_coco.yaml`: The config for loading the two-label dataset
- `data/custom_coco.yaml`: The config for loading the single-label dataset
- `train.py`: Modified for training
- `test.py`: Modified for post-processing the output of NMS and pre-processing the input of NMS for detecting the smallest and largest objects.
- `detect.py`: Modified for post-processing the output of NMS and pre-processing the input of NMS for detecting the smallest and largest objects.
## 4. Using the code
For pre-process the raw data, run the following code
```
# Create the dataset for singe-object-detection training
python utils/preprocess_single_class.py --output_dir dataset/processed/single_class

# Create the dataset for multiple-object-detection training
python utils/preprocess_two_classes.py --output_dir dataset/processed/two_class/

# Train for detecting single class
python train.py --workers 8 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights pretrained/yolov7.pt --name tmp --hyp data/hyp.scratch.p5.yaml --single-cls

# Train for detecting multiple classes
python train.py --workers 8 --batch-size 32 --data data/custom_coco_two.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights pretrained/yolov7.pt --name tmp --hyp data/hyp.scratch.p5.yaml

# Run test.py to compute the test statistics on the validation set
python test.py --weights runs/train/singleclass_yolo_freeze5/weights/best.pt --data data/custom_coco.yaml
```
