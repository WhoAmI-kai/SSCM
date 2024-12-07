import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader, DataListLoader, Batch, Data
from argoverse.map_representation.map_api import ArgoverseMap
from frenet import *
from core.model.vectornet import VectorNet
from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem

# def viz(data, centerline, dataset_input_path):
#     # am = ArgoverseMap()
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     # index = data.seq_id.cpu().clone().detach().numpy()
#     # path = os.path.join(dataset_input_path, f'raw/features_{str(index[0])}.pkl')
#     # pkl_data = pd.read_pickle(path)
#     # print(pkl_data['trajs'][0][0][0:20])
#     # agt_traj_obs = pkl_data['trajs'][0][0][0: 20].copy().astype(np.float32)
#     # print(agt_traj_obs)
#     # print(pkl_data['city'])
#     # ctr_line_candts = am.get_candidate_centerlines_for_traj(agt_traj_obs, pkl_data['city'].values[0], viz=False)
#
#     # orig = data.orig.cpu().clone().detach().numpy().reshape(-1)
#     rot = data.rot.cpu().clone().detach().numpy().reshape(2, 2)
#
#     y = data.y.clone().numpy().reshape((-1, 2)).cumsum(axis=0)
#     y = np.matmul(np.linalg.inv(rot), y.T).T
#
#     ax.plot(y[:, 0], y[:, 1])
#     ax.plot(centerline[:, 0], centerline[:, 1])
#
#     # for ctr in ctr_line_candts:
#     #     ax.plot(ctr[:, 0] - orig[0], ctr[:, 1] - orig[1])
#
#     fig.show()
#     plt.close(fig)

if __name__ == '__main__':
    INTERMEDIATE_DATA_DIR = './scene/interm_data/'
    folder = 'tmp'
    dataset_input_path = os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")
    dataset = ArgoverseInMem(dataset_input_path)
    data_iter = DataLoader(dataset, batch_size=1, num_workers=16, pin_memory=True)

    in_channels, pred_len = 10, 30
    model = VectorNet(in_channels, pred_len, with_aux=True)
    state = torch.load('/data/yingkai/Attack/trained_params/best_VectorNet.pth', map_location='cuda:0')
    model.load_state_dict(state)
    model.eval()

    for j, data in enumerate(data_iter):
        centerlane = reference_line(data, dataset_input_path)
        centerlane_dis = []
        for j in range(centerlane.shape[0]):
            if j == 0:
                centerlane_dis.append(DiscretizedReferenceLine(j, centerlane))
            else:
                centerlane_dis.append(DiscretizedReferenceLine(j, centerlane, centerlane_dis[j-1]))


        fig, ax = plt.subplots(figsize=(10, 10))
        xy = data.x[:19, :2]
        ax.plot(xy[:, 0], xy[:, 1])
        fig.savefig('./scene/interm_data/tmp_intermediate/vis/{}.png'.format(j))
        plt.close(fig)


