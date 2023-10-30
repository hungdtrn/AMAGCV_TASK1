# Report for Computer Vision Task: Detecting the largest and the smallest objects in Images / Videos using YOLO.
This report outlines my approaches to fine-tune the YOLO-V7 model for detecting the largest and smallest objects in images and videos.

## 1. Summary
### 1.1. About the data
• There is no clear distinction between smallest and largest bounding boxes. Sizes alone cannot determine this distinction.
• Some objects are exceptionally small (as small as 0.8 pixels), making accurate labeling difficult. They were removed from the data. 
### 1.2. About the models
• Fine-tuning, then post-processing: Two variants of this approach are considered: (1) Fine-tuning to detect largest and smallest boxes direct from the image (2) Fine-tune the model to detect objects in the image without distinguishing between specific objects, and subsequently select two boxes representing the smallest and largest sizes.
• Regardless of the model, an additional post-processing step is necessary for ensuring the model only output a largest and a smallest object in the image. 
## 2. Preprequisite Installation
Install the environment
```
conda env create -f environment.yml
conda activate amagcvTask1
```
Install and setup wandb as instructed [here](https://docs.wandb.ai/quickstart)

Download the dataset and model and unzip to the desired folder
```
bash download.sh 
```
## 3. Models
### Model list and pretrained weights

Model | Precision | Recall | mAP@50 | Link
--- | --- | --- | --- | ---
Two-class, 3 anchors | 0.500 | 0.460 | 0.360 | [Link](https://drive.google.com/file/d/1H0wpSU3D8OTylmewas5tS9ysxVD9X5C7/view?usp=drive_link)
Two-class, 5 anchors | 0.546 | 0.483 | 0.3910 | [Link](https://drive.google.com/file/d/1ATLpKHtDHkf3Tr693UfTNrL7Of_oNRyQ/view?usp=drive_link)
Single-class, 3 anchors | 0.612 | 0.474 | 0.435 | [Link](https://drive.google.com/file/d/1cDrzFMmPECClK9h9KC8zYLlfATJbq_fL/view?usp=drive_link)
Single-class, 5 anchors | 0.665 | 0.486 | 0.457 | [Link](https://drive.google.com/file/d/1aGPCqJ6ipOXa6RDXfrD-JMugpI1V38hb/view?usp=drive_link)

Two-class: Fine-tune the model to identify the largest and smallest object boxes within the image, then select the boxes that have the smallest and largest size.
Single-class: Fine-tune the model to detect all objects in the image, then select two boxes that have smallest and largest size.
## 4. Using the code
### Pre-processing the raw  data
```
python utils/prepare_dataset.py --output_dir dataset/processed/ --min_size 8
```

The data used to trained and evaluated the models were uploaded [here](https://drive.google.com/file/d/1f-yonKKwXrHFc5elUprzIwR6TDh5En2m/view?usp=drive_link). Please download it and unzip to the `dataset` folder.

### Training models
Two-class model
```
python train.py --workers 8 --batch-size 32 --data data/single_class_coco.yaml --img 640 640 --cfg cfg/training/yolov7_5anchors.yaml --weights downloaded_files/yolov7.pt  --name singleClass5anchor  --hyp data/hyp.scratch.p5.yaml --device 0
```

Single-class model
```
python train.py --workers 8 --batch-size 32 --data data/single_class_coco.yaml --img 640 640 --cfg cfg/training/yolov7_5anchors.yaml --weights downloaded_files/yolov7.pt  --name singleClass5anchor  --hyp data/hyp.scratch.p5.yaml --device 0
```
### Evaluating the model
```
python test.py --batch-size 32 --data data/single_class_coco.yaml  --weights model.pt --device 0
```
Where `model.pt` is the path to your model
### Apply on some other images
```
python detect.py --weight model.pt  --device 0 --img-size 640 --source inference/images/
```
### Some Demo
![Single-class, 5 anchors](/public/singleclass.jpg)
![Single-class, 5 anchors](/public/singleclass1.jpg)

## Report
My report is uploaded [Here](https://drive.google.com/file/d/1oup1Vg27Hom9GQ_5CPw2LniBchlQpieq/view?usp=sharing)


