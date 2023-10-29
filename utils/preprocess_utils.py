from collections import defaultdict

def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def filter_invalid_boxes(boxList, image_info, min_size):
    ninvalid = 0
    nobjs = 0
    imgw, imgh = image_info["size"]
    new_boxList = []
    ratio = image_info["ratio"]
    for b in boxList:
        _, _, bw, bh = b["bbox"]

        nobjs += 1
        
        # Check if th boxes are bigger than the image
        if bw > imgw or bh > imgh or bw < 0 or bh < 0:
            ninvalid += 1
            continue

        # Check if the size of the boxes are still valid after resizing the image
        bw *= ratio
        bh *= ratio
        if min_size is not None and min(bw, bh) < min_size: 
            ninvalid += 1           
            continue
        
        new_boxList.append(b)
    
    return new_boxList, ninvalid, nobjs
    

def select_two_objects(box_list, image_info, min_smallest, max_smallest, min_largest, max_largest,
                       min_size):
    min_area, max_area = float("inf"), -float("inf")
    min_box, max_box = None, None
    ratio = image_info["ratio"]
    for b in box_list: 
        _, _, bw, bh = b["bbox"]
               
        bw *= ratio
        bh *= ratio
        
        area = bw * bh

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

def preprocess_boxes(annotations, image_info, min_smallest,
                     max_smallest, min_largest, max_largest,
                     min_size,
                     num_objects=2):
    boxPerImage = defaultdict(list)

    for annotation in annotations["annotations"]:
        boxPerImage[annotation["image_id"]].append(annotation)
        
    tmp = {}
    
    total_invalid, total_obj = 0, 0
    for k in boxPerImage:
        boxes, ninvalid, nobjs = filter_invalid_boxes(boxPerImage[k], image_info[k],
                                     min_size=min_size)
        total_invalid += ninvalid   
        total_obj += nobjs
        
        if num_objects == 2:
            boxes = select_two_objects(boxes, image_info[k],
                                    min_smallest, max_smallest, min_largest, max_largest,
                                    min_size=min_size)
            
        if boxes is not None:
            tmp[k] = boxes
            
    boxPerImage = tmp
    print("Total invalid", total_invalid, "Total Obj", total_obj, "Ratio", total_invalid/total_obj)
    return boxPerImage