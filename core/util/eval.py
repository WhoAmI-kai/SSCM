#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-06-18 22:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
import torch
from tqdm import tqdm
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from pprint import pprint
from typing import List


def get_eval_metric_results(model, data_loader, device, horizon, miss_threshold):
    """
    ADE, FDE, and Miss Rate
    """
    forecasted_trajectories, gt_trajectories = {}, {}
    seq_id = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            batch_size = data.num_graphs
            gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()
            out = model.inference(data.to(device))
            dim_out = len(out.shape)
            pred_y = out.unsqueeze(dim_out).view((batch_size, 1, horizon, 2)).cpu().numpy()

            for batch_id in range(batch_size):
                forecasted_trajectories[seq_id] = [pred_y_k for pred_y_k in pred_y[batch_id]]
                gt_trajectories[seq_id] = gt[batch_id]
                seq_id += 1
            # print(out)

        metric_results = get_displacement_errors_and_miss_rate(
            forecasted_trajectories,
            gt_trajectories,
            1,
            horizon,
            miss_threshold
        )
    return metric_results

    #     gt = None
    #     # mutil gpu testing
    #     if isinstance(data, List):
    #         gt = torch.cat([i.y for i in data], 0).view(-1, out_channels).to(device)
    #     # single gpu testing
    #     else:
    #         data = data.to(device)
    #         gt = data.y.view(-1, out_channels).to(device)
    #
    #     out = model(data)
    #     for i in range(gt.size(0)):
    #         pred_y = out[i].view((-1, 2)).cumsum(axis=0).cpu().numpy()
    #         y = gt[i].view((-1, 2)).cumsum(axis=0).cpu().numpy()
    #         forecasted_trajectories[seq_id] = [pred_y]
    #         gt_trajectories[seq_id] = y
    #         seq_id += 1
    # metric_results = get_displacement_errors_and_miss_rate(
    #     forecasted_trajectories, gt_trajectories, max_n_guesses, horizon, miss_threshold
    # )
#         return metric_results
#
# # def eval_loss():
#     raise NotImplementedError("not finished yet")
#     model.eval()
#     from utils.viz_utils import show_pred_and_gt
#     with torch.no_grad():
#         accum_loss = .0
#         for sample_id, data in enumerate(train_loader):
#             data = data.to(device)
#             gt = data.y.view(-1, out_channels).to(device)
#             optimizer.zero_grad()
#             out = model(data)
#             loss = F.mse_loss(out, gt)
#             accum_loss += batch_size * loss.item()
#             print(f"loss for sample {sample_id}: {loss.item():.3f}")
#
#             for i in range(gt.size(0)):
#                 pred_y = out[i].numpy().reshape((-1, 2)).cumsum(axis=0)
#                 y = gt[i].numpy().reshape((-1, 2)).cumsum(axis=0)
#                 show_pred_and_gt(pred_y, y)
#                 plt.show()
#         print(f"eval overall loss: {accum_loss / len(ds):.3f}")
