#!/bin/bash

echo "Create a foler for storing downloaded files"
mkdir ./downloaded_files

echo "Download files"
wget http://images.cocodataset.org/zips/val2017.zip -P ./downloaded_files
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./downloaded_files
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -P ./downloaded_files

echo "Extract files"
mkdir dataset 
mkdir dataset/raw
mkdir dataset/processed
unzip ./downloaded_files/val2017.zip -d ./dataset/raw
mv ./dataset/raw/val2017/ ./dataset/raw/images
unzip ./downloaded_files/annotations_trainval2017.zip -d ./dataset/raw