import os
import re
import time

import torch
from torch_geometric.data import DataLoader, DataListLoader, Batch, Data

from frenet import *
from attack_core_right import Attack
from viz_new_model import *
from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem
from core.model.vectornet import VectorNet


def generate_data(data, samples=1, targeted = True):
    inputs = []
    ground_truths = []
    targets = []

    for i in range(samples):
        inputs.append(data.x)
        ground_truths.append(data.y)
        if targeted:
            targets = np.array([-2.3])

    return inputs, ground_truths, targets

if __name__ == "__main__":
    in_channels, pred_len = 10, 30
    batch_size = 1
    #
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder = 'filter_2'
    dataset_input_path = os.path.join(
        './filter/interm_data', f"{folder}_intermediate")

    dataset = ArgoverseInMem(dataset_input_path)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    model = VectorNet(in_channels, pred_len, with_aux=True).to(device)
    state = torch.load('./trained_params/best_VectorNet.pth', map_location='cuda:0')
    model.load_state_dict(state)

    path_conclusion = r'./tools/analy_filter.txt'
    name_list = []
    with open(path_conclusion, 'r') as f:
        for line in f:
            if 'false' in line:
                name = re.findall(r'\d+', line)
                name_list.append(int(name[0]))
    name_list = np.array(name_list).reshape(-1)

    file_num = 0
    for i, data in enumerate(data_iter):
        # if data.seq_id.item() == 39285:
        if np.isin(data.seq_id.item(), name_list):
            continue
        centerlane = reference_line(data, dataset_input_path)
        centerlane_dis = []
        for j in range(centerlane.shape[0]):
            if j == 0:
                centerlane_dis.append(DiscretizedReferenceLine(j, centerlane))
            else:
                centerlane_dis.append(DiscretizedReferenceLine(j, centerlane, centerlane_dis[j-1]))

        inputs, ground_truths, targets = generate_data(data)
        # attack = Attack(model, dataset=data, agt_num=19, centerlane_dis=centerlane_dis, max_iterations=10000,
        #                 initial_const=10, largest_const=1e4, initial_w1=1e3, largest_w1=1e5)
        attack = Attack(model, dataset=data, agt_num=19, centerlane_dis=centerlane_dis, max_iterations=10000,
                        initial_const=20, largest_const=1e4, initial_w1=800, largest_w1=1e5)

        timestart = time.time()
        adv = attack.attack(inputs, ground_truths, targets)
        timeend = time.time()

        print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")
        # print("\nadv:\n", adv)
        seq_id = data.seq_id.item()
        ae_path = './AE_filter_right/AE/' + str(seq_id) + '.pt'
        torch.save(adv, ae_path)
        if file_num == 99:
            break
        else:
            file_num += 1
        # viz(data, centerlane, dataset_input_path)
        # break