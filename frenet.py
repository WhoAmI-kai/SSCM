import os
import torch
import math
import numpy as np
import pandas as pd
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

am = ArgoverseMap()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# norm_center_all = pd.read_pickle('../interm_data/train-norm_center_dict.pkl')
# city_name_all = pd.read_pickle('../interm_data/train-city_name_dict.pkl')
# # norm_center = norm_center_all['1000']
# data_dir = '../interm_data/train_intermediate'
# data_path_ls = sorted([os.path.join(data_dir, data_path) for data_path in os.listdir(data_dir)])
#
# def get_norm_center(index):
#     key = "".join(list(filter(str.isdigit, os.path.basename(data_path_ls[index]))))
#     return norm_center_all[str(key)]

def get_city_name(index, dataset_input_path):
    # key = "".join(list(filter(str.isdigit, os.path.basename(data_path_ls[index]))))
    # return city_name_all[str(key)]
    index = str(index[0])
    path = os.path.join(dataset_input_path, f'raw/features_{index}.pkl')
    pkl_data = pd.read_pickle(path)
    city_name = pkl_data['city'].values[0]
    return city_name


def reference_line(data, dataset_input_path):
    """
    最近的中心线，只对原数据计算，adv仍使用该中心线
    Args:
        data: ground truth, type=torch.tensor

    Returns:
        nearest_centerlane_norm: 最近的中心线， type=np.array
    """

    ground_truth = data.y.cpu().clone().detach().numpy().reshape((-1, 2)).cumsum(axis=0)
    ground_truth_last_point = ground_truth[-1, :2]

    norm_center = data.orig.cpu().clone().detach().numpy().reshape(-1)
    seq_id = data.seq_id.cpu().clone().detach().numpy()
    rot = data.rot.cpu().clone().detach().numpy().reshape(2, 2)

    query_xy = ground_truth_last_point.reshape(-1)
    # print(query_xy_norm)
    # print(norm_center)
    query_xy = np.matmul(np.linalg.inv(rot), query_xy.T).T + norm_center

    city_name = get_city_name(seq_id, dataset_input_path)
    nearest_centerlane = am.get_nearest_centerline(query_xy, city_name)
    nearest_centerlane_norm = nearest_centerlane[2] - norm_center

    return nearest_centerlane_norm


# def newvec_reference(data, agt_num):
#     newvec = data.clone().cpu().detach().numpy()
#     newvec_last_point = newvec[agt_num, :2]
#
#     query_xy = newvec_last_point + norm_center
#
#     nearest_centerlane = am.get_nearest_centerline(query_xy, "MIA")
#     nearest_centerlane_norm = nearest_centerlane[2] - norm_center
#     nearest_centerlane_torch = torch.from_numpy(nearest_centerlane_norm)
#
#     return nearest_centerlane_torch


# def frenet_l(output_data, nearest_centerlane_norm):
#     """
#     point在frenet坐标系中的l坐标
#     Args:
#         output_data: 对抗样本的预测值，type=torch.tensor
#         nearest_centerlane_norm: type=torch.tensor
#
#     Returns:
#
#     """
#     point_all = torch.cumsum(output_data.reshape(-1, 2))
#     point = point_all[-1, :2]
#     dist = torch.zeros(nearest_centerlane_norm.shape[0]).to(device)
#     for i in range(nearest_centerlane_norm.shape[0]):
#         dist[i] = torch.dist(point, nearest_centerlane_norm[i, :2], p=2)
#         i += 1
#     dist_min_index = torch.argmin(dist)
#
#     e = nearest_centerlane_norm[dist_min_index - 1]
#     f = nearest_centerlane_norm[dist_min_index + 1]
#     ea = point - e
#     ef = f - e
#     s = torch.dot(ea, ef) / torch.norm(ef, p=2)
#     l = (torch.norm(ea, p=2) ** 2 - s ** 2) ** 0.5
#
#     return l

def NormalizeAngle(angle):
    a = (angle + math.pi) % (2 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def slerp(a0, t0, a1, t1, t):
    a0_n = NormalizeAngle(a0)
    a1_n = NormalizeAngle(a1)
    d = a1_n - a0_n
    if d > math.pi:
        d = d - 2 * math.pi
    elif d < -math.pi:
        d = d + 2 * math.pi
    r = (t - t0) / (t1 - t0)
    a = a0_n + d * r
    return NormalizeAngle(a)


def MatchToPath(data, centerlane_dis, no_need_cuda=False):
    """
    point在frenet坐标系中的l坐标
    Args:
        output_data: initial point，type=torch.tensor
        centerlane_dis: centerlane中每个点的类构成的列表

    Returns:
        参考线上匹配点的 s, x, y, theta

    """
    # rot_inv = torch.from_numpy(np.linalg.inv(rot.cpu().clone().detach().numpy().reshape(2, 2))).to(device)
    # point_all = torch.cumsum(output_data.reshape(-1, 2), dim=0)
    # point = point_all[-1, :2]
    # point = torch.matmul(rot_inv, point.T).T
    if no_need_cuda == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(len(centerlane_dis))
    dist = torch.zeros(len(centerlane_dis)).to(device)
    for i in range(len(centerlane_dis)):
        center_torch = torch.from_numpy(centerlane_dis[i].xy).to(device)
        dist[i] = torch.dist(data, center_torch, p=2)
    dist_min_index = torch.argmin(dist)

    index_start = dist_min_index - 1 if dist_min_index != 0 else dist_min_index
    index_end = dist_min_index + 1 if dist_min_index != len(centerlane_dis) - 1 else dist_min_index

    assert index_start != index_end, 'centerlane is a point'

    p0 = centerlane_dis[index_start]
    p1 = centerlane_dis[index_end]

    p0_xy = torch.from_numpy(p0.xy).to(device)
    p1_xy = torch.from_numpy(p1.xy).to(device)
    v0 = data - p0_xy
    v1 = p1_xy - p0_xy

    delta_s = torch.dot(v0, v1) / torch.norm(v1, p=2)
    s = p0.s + delta_s

    # 一阶线性插值算匹配点的 x, y, theta
    s0 = p0.s
    s1 = p1.s
    weight = (s - s0) / (s1 - s0)
    x = (1 - weight) * p0.x + weight * p1.x
    y = (1 - weight) * p0.y + weight * p1.y
    theta = slerp(p0.theta, p0.s, p1.theta, p1.s, s)
    return s, x, y, theta


def cartesian_frenet(rs, rx, ry, rtheta, data, no_need_cuda=False):
    """
    Args:
        rs,rx,ry,rtheta：参考线上匹配点的s,x,y,theta
        x,y: initial point 的 x 和 y
    Returns:
        initial point 在 Frenet坐标系下的 l

    """
    # # 向量AB的分量
    # rot_inv = torch.from_numpy(np.linalg.inv(rot.cpu().clone().detach().numpy().reshape(2, 2))).to(device)
    # point_all = torch.cumsum(data.reshape(-1, 2), dim=0)
    # point = point_all[-1, :2]
    # point = torch.matmul(rot_inv, point.T).T
    if no_need_cuda is True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = data[0]
    y = data[1]
    dx = x - rx
    dy = y - ry

    # 过A点的单位切向量
    cos_theta_r = torch.cos(rtheta).to(device)
    sin_theta_r = torch.sin(rtheta).to(device)

    # AB的方向
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    l_undirect = (dx * dx + dy * dy) ** 0.5
    l = torch.copysign(l_undirect, cross_rd_nd)
    return l


def frenet_cartesian(rx, ry, rtheta, l):
    xy = torch.zeros([2]).to(device)
    x = rx - torch.sin(rtheta) * l
    y = ry + torch.cos(rtheta) * l
    xy[0] = x
    xy[1] = y
    return xy


class DiscretizedReferenceLine:
    def __init__(self, n, centerlane, centerlane_n0=None):
        self.n = n
        self.centerlane = centerlane
        self.x = centerlane[n, 0]
        self.y = centerlane[n, 1]
        self.xy = np.array([self.x, self.y])
        self.centerlane_n0 = centerlane_n0
        self.s = self.get_s()
        self.theta = self.get_theta()

    def get_s(self):
        if self.n == 0:
            return 0
        else:
            dx = self.x - self.centerlane_n0.x
            dy = self.y - self.centerlane_n0.y
            s = self.centerlane_n0.s + (dx * dx + dy * dy) ** 0.5
            return s

    def get_theta(self):
        if self.n == 0:
            v0_x = self.centerlane[1, 0] - self.x
            v0_y = self.centerlane[1, 1] - self.y
        else:
            v0_x = self.x - self.centerlane_n0.x
            v0_y = self.y - self.centerlane_n0.y
        if v0_x == 0 and v0_y == 0:
            theta = 0
        elif v0_y >= 0:
            theta = np.arccos(v0_x / ((v0_x * v0_x + v0_y * v0_y) ** 0.5))
        else:
            theta = 2 * math.pi - np.arccos(v0_x / ((v0_x * v0_x + v0_y * v0_y) ** 0.5))
        return theta


class DiscretizedReferenceLine_newvec:
    def __init__(self, n, centerlane, centerlane_n0=None):
        self.n = n
        self.centerlane = centerlane
        self.x = centerlane[n, 0]
        self.y = centerlane[n, 1]
        self.xy = torch.tensor([self.x, self.y]).to(device)
        self.centerlane_n0 = centerlane_n0
        self.s = self.get_s()
        self.theta = self.get_theta()

    def get_s(self):
        if self.n == 0:
            return 0
        else:
            dx = self.x - self.centerlane_n0.x
            dy = self.y - self.centerlane_n0.y
            s = self.centerlane_n0.s + (dx * dx + dy * dy) ** 0.5
            return s

    def get_theta(self):
        if self.n == 0:
            v0_x = self.centerlane[1, 0] - self.x
            v0_y = self.centerlane[1, 1] - self.y
        else:
            v0_x = self.x - self.centerlane_n0.x
            v0_y = self.y - self.centerlane_n0.y
        if v0_x == 0 and v0_y == 0:
            theta = torch.zeros(1).to(device)
        elif v0_y >= 0:
            theta = torch.arccos(v0_x / ((v0_x * v0_x + v0_y * v0_y) ** 0.5))
        else:
            theta = 2 * math.pi - torch.arccos(v0_x / ((v0_x * v0_x + v0_y * v0_y) ** 0.5))
        return theta


class newvec_frenet:
    def __init__(self, n, data, centerlane_dis):
        self.n = n
        self.x = data[self.n, 0]
        self.y = data[self.n, 1]
        self.xy = data[self.n, :2]
        self.centerlane_dis = centerlane_dis
        self.mp_s, self.mp_x, self.mp_y, self.mp_theta = self.match_point()
        self.l = self.cartesian_frenet()

    def match_point(self):
        dist = torch.zeros(len(self.centerlane_dis)).to(device)
        for i in range(len(self.centerlane_dis)):
            center_torch = self.centerlane_dis[i].xy.to(device)
            dist[i] = torch.dist(self.xy, center_torch, p=2)
            i += 1
        dist_min_index = torch.argmin(dist)

        index_start = dist_min_index - 1 if dist_min_index != 0 else dist_min_index
        index_end = dist_min_index + 1 if dist_min_index != len(self.centerlane_dis) else dist_min_index

        assert index_start != index_end, 'centerlane is a point'

        p0 = self.centerlane_dis[index_start]
        p1 = self.centerlane_dis[index_end]

        v0 = self.xy - p0.xy
        v1 = p1.xy - p0.xy

        delta_s = torch.dot(v0, v1) / torch.norm(v1, p=2)
        s = p0.s + delta_s

        # 一阶线性插值算匹配点的 x, y, theta
        s0 = p0.s
        s1 = p1.s
        weight = (s - s0) / (s1 - s0)
        x = (1 - weight) * p0.x + weight * p1.x
        y = (1 - weight) * p0.y + weight * p1.y
        theta = slerp(p0.theta, p0.s, p1.theta, p1.s, s)
        return s, x, y, theta

    def cartesian_frenet(self):
        dx = self.x - self.mp_x
        dy = self.y - self.mp_y

        # 过A点的单位切向量
        cos_theta_r = torch.cos(self.mp_theta).to(device)
        sin_theta_r = torch.sin(self.mp_theta).to(device)

        # AB的方向
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        l_undirect = (dx * dx + dy * dy) ** 0.5
        l = torch.copysign(l_undirect, cross_rd_nd)
        return l


class modifier_transform:
    def __init__(self, n, modifier, raw_point, per_point):
        self.n = n
        self.modifier = modifier
        self.raw_point = raw_point
        self.per_point = per_point
        self.modi_trans = self.trans()

    def trans(self):
        # 原始点为A，扰动点为B在AC上的映射为C
        rx = self.raw_point.mp_x
        ry = self.raw_point.mp_y
        rtheta = self.raw_point.mp_theta
        # if self.per_point.l - self.raw_point.l >= 0:
        #     l = self.raw_point.l + 1
        # else:
        #     l = self.raw_point.l - 1
        # xy = frenet_cartesian(rx, ry, rtheta, l)

        v0 = self.per_point.xy - torch.tensor([self.raw_point.mp_x, self.raw_point.mp_y]).to(device)
        v1 = self.raw_point.xy - torch.tensor([self.raw_point.mp_x, self.raw_point.mp_y]).to(device)

        AD = torch.dot(v0, v1) / torch.norm(v1, p=2)
        l_trans = torch.copysign(AD, self.per_point.l)

        D_xy = frenet_cartesian(rx, ry, rtheta, l_trans)

        return D_xy
