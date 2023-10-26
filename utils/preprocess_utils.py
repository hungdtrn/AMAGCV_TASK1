from collections import defaultdict

def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def select_two_objects(box_list, min_smallest, max_smallest, min_largest, max_largest):
    min_area, max_area = float("inf"), -float("inf")
    min_box, max_box = None, None
    for b in box_list:
        
        area = b["bbox"][-1] * b["bbox"][-2]
        if area < min_area:
            if min_smallest is not None and area < min_smallest:
                continue
            if max_smallest is not None and area > max_smallest:
                continue
            
            min_area = area
            min_box = b
        
        if area > max_area:
            if min_largest is not None and area < min_largest:
                continue
            if max_largest is not None and area > max_largest:
                continue
            
            max_area = area
            max_box = b
    
    if min_area == max_area or min_area == float("inf") or max_area == -float("inf"):
        return None
    
    return [min_box, max_box]

def preprocess_boxes(annotations, min_smallest,
                     max_smallest, min_largest, max_largest):
    boxPerImage = defaultdict(list)

    for annotation in annotations["annotations"]:
        boxPerImage[annotation["image_id"]].append(annotation)
    
    print(len(boxPerImage))
    
    tmp = {}
    for k in boxPerImage:
        boxes = select_two_objects(boxPerImage[k],
                                   min_smallest, max_smallest, min_largest, max_largest)
        if boxes is not None:
            tmp[k] = boxes
            
    
    boxPerImage = tmp
    return boxPerImage