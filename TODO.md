Requirement:
1. Download the data & annotations. 
2. Prepocessing the data  
3. Change the architecture to only detect the biggest and the smallest objects.

## Processing the data
The number of all data is 5000.

### Select the biggest and the smallest objects 

### Train-test split
val set: 10%
Train set: 90%

We need to ensure that the distribution of the bounding box classes are all most the same between the train and the val set.

The label format is 
class_id x, y, h, w

Filter the data that has only one label. 




## Possible solutions:
### Solution 1: Changing anchors
- Compute the new set of anchors for the new dataset
- Train the model with the new anchor

### Solution 2: Filetering candiates
- 
