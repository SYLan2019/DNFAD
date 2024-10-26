import numpy as np
import torch
import torch.nn.functional as F


def post_process_first(c, outputs):
    outputs = torch.cat(outputs, 0)
    logp_map = F.interpolate(outputs.unsqueeze(1),
                             size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
    logp_map -= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    prob_map_mul = torch.exp(logp_map)
    anomaly_score_map = prob_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prob_map_mul
    batch = anomaly_score_map.shape[0]
    top_k = int(c.crop_size[0] * c.crop_size[1] * c.top_k)
    anomaly_score = np.mean(
        anomaly_score_map.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)
    output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    prob_map = torch.exp(output_norm)

    prop_map = F.interpolate(prob_map.unsqueeze(1),
                             size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
    prop_map = prop_map.detach().cpu().numpy()
    anomaly_score_map_loc = prop_map.max(axis=(1, 2), keepdims=True) - prop_map

    return anomaly_score, anomaly_score_map_loc, anomaly_score_map.detach().cpu().numpy()

def post_process(c, outputs_list):
    anomaly_score_patch_level, anomaly_score_map_loc_patch_level, anomaly_score_map_patch_level = post_process_first(c,
                                                                                                                     outputs_list[
                                                                                                                         0])
    anomaly_score_map_level, anomaly_score_map_loc_map_level, anomaly_score_map_map_level = post_process_first(c,
                                                                                                               outputs_list[
                                                                                                                   1])
    # two branch
    anomaly_score = anomaly_score_map_level + anomaly_score_patch_level
    anomaly_score_map = anomaly_score_map_loc_map_level + anomaly_score_map_loc_patch_level
    anomaly_score_map_level = anomaly_score_map_map_level + anomaly_score_map_patch_level

    return anomaly_score, anomaly_score_map, anomaly_score_map_level
