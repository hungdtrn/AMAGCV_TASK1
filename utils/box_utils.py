import torch 
def preprocess_boxes(out, conf_thresh=0.1, min_area_bound=None):
    area = out[:, :, 2:4].sum(axis=2)
    candidate = (out[:, :, 4] > conf_thresh) 

    if out.shape[-1] == 6:
        min_area = torch.where(candidate, area, torch.max(candidate)).min(axis=1, keepdim=True).values
        max_area = torch.where(candidate, area, 0).max(axis=1, keepdim=True).values
        flag = (area == min_area) | (area == max_area)
        out[:,  :, 4] = out[:,  :, 4] * flag.float()
        out = torch.cat([out[:, :, :5], (area == min_area).unsqueeze(-1), (area == max_area).unsqueeze(-1)], dim=-1)
    elif out.shape[-1] == 7:
        smallest_score = out[:, :, 5:6]
        largest_score = out[:, :, 6:7]
        smallest_flag = smallest_score > largest_score
        largest_flag = largest_score >= smallest_score
        
        smallest_candidate = candidate & smallest_flag.squeeze(-1)
        largest_candidate = candidate & largest_flag.squeeze(-1)
        min_area = torch.where(smallest_candidate, area, 1000).min(axis=1, keepdim=True).values
        max_area = torch.where(largest_candidate, area, 0).max(axis=1, keepdim=True).values
        
        flag = (area == min_area) | (area == max_area)
        out[:,  :, 4] = out[:,  :, 4] * flag.float()
    else:
        min_area = torch.where(candidate, area, 1000).min(axis=1, keepdim=True).values
        max_area = torch.where(candidate, area, 0).max(axis=1, keepdim=True).values
        flag = (area == min_area) | (area == max_area)
        out[:,  :, 4] = out[:,  :, 4] * flag.float()
        out = torch.cat([out[:, :, :5], (area == min_area).unsqueeze(-1), (area == max_area).unsqueeze(-1)], dim=-1)
        
    return out

def postprocess_boxes(out, conf_thresh=0.1, min_area_bound=None,
                      min_edge_bound=None, topk=1):
    new_output = []
    for x in out:
        if not len(x):
            new_output.append(x)
            continue

        if topk > len(x):
            topk = 1
            
        w, h = x[:, 2] - x[:,0], x[:, 3] - x[:, 1]

        area = w * h
        

        # min_flag = (x[:, 4] > 0.1) & torch.all(x[:, :4] > 0, dim=1)
        # max_flag = x[:, 4] > 0.3
        # min_area_id = torch.topk(torch.where(min_flag, area, torch.max(area)), topk, dim=0, largest=False)[1]
        # max_area_id = torch.topk(torch.where(max_flag, area, torch.min(area)), topk, dim=0, largest=True)[1]

        # ratio = area / (torch.max(area) + 1e-5)
        # min_area_id = torch.topk(torch.exp(ratio + (1 - x[:, 4])), topk, dim=0, largest=False)[1]
        # max_area_id = torch.topk(torch.exp(ratio + x[:, 4]), topk, dim=0, largest=True)[1]
        
        min_area_id = torch.topk(area, topk, dim=0, largest=False)[1]
        max_area_id = torch.topk(area, topk, dim=0, largest=True)[1]
        
        box_min = x[min_area_id]
        box_max = x[max_area_id]
                        
        box_min = torch.cat([box_min[:, :5], torch.zeros_like(box_min[:, 4:5])], dim=-1)
        box_max = torch.cat([box_max[:, :5], torch.ones_like(box_max[:, 4:5])], dim=-1)
        
        new_output.append(torch.cat([box_min, box_max]))

    out = new_output

    return out

