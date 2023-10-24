import torch 
def preprocess_boxes(out, conf_thresh=0.1):
    print("Preprocessing the candidate")

    circumference = out[:, :, 2:4].sum(axis=2)
    min_allowed_circumference = 3 

    candidate = (out[:, :, 4] > conf_thresh) & (circumference > min_allowed_circumference)

    if out.shape[-1] == 6:
        min_circumference = torch.where(candidate, circumference, 1000).min(axis=1, keepdim=True).values
        max_circumference = torch.where(candidate, circumference, 0).max(axis=1, keepdim=True).values
        flag = (circumference == min_circumference) | (circumference == max_circumference)
        out[:,  :, 4] = out[:,  :, 4] * flag.float()
        out = torch.cat([out[:, :, :5], (circumference == min_circumference).unsqueeze(-1), (circumference == max_circumference).unsqueeze(-1)], dim=-1)
    elif out.shape[-1] == 7:
        smallest_score = out[:, :, 5:6]
        largest_score = out[:, :, 6:7]
        smallest_flag = smallest_score > largest_score
        largest_flag = largest_score >= smallest_score
        
        smallest_candidate = candidate & smallest_flag.squeeze(-1)
        largest_candidate = candidate & largest_flag.squeeze(-1)
        min_circumference = torch.where(smallest_candidate, circumference, 1000).min(axis=1, keepdim=True).values
        max_circumference = torch.where(largest_candidate, circumference, 0).max(axis=1, keepdim=True).values
        
        flag = (circumference == min_circumference) | (circumference == max_circumference)
        out[:,  :, 4] = out[:,  :, 4] * flag.float()
    else:
        min_circumference = torch.where(candidate, circumference, 1000).min(axis=1, keepdim=True).values
        max_circumference = torch.where(candidate, circumference, 0).max(axis=1, keepdim=True).values
        flag = (circumference == min_circumference) | (circumference == max_circumference)
        out[:,  :, 4] = out[:,  :, 4] * flag.float()
        out = torch.cat([out[:, :, :5], (circumference == min_circumference).unsqueeze(-1), (circumference == max_circumference).unsqueeze(-1)], dim=-1)
        
    return out

def postprocess_boxes(out, conf_thresh=0.1):
    print("Postprocessing the boxes")
    new_output = []
    for x in out:
        if not len(x):
            new_output.append(x)
            continue

        w, h = x[:, 2] - x[:,0], x[:, 3] - x[:, 1]

        area = w * h
        min_area_id, max_area_id = torch.argmin(area), torch.argmax(area)        
        box_min = x[min_area_id]
        
        box_max = x[max_area_id]
                
        box_min = torch.cat([box_min[:5], torch.zeros_like(box_min[4:5])], dim=-1)
        box_max = torch.cat([box_max[:5], torch.ones_like(box_min[4:5])], dim=-1)
        
        new_output.append(torch.stack([box_min, box_max]))

    out = new_output

    return out

