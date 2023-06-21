# coding=utf-8
import time
import numpy as np
import pandas
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform


def compute_dense(data: np.ndarray, point: np.ndarray, r: float) -> np.ndarray:
    nn = data.shape[0]
    dd = data.shape[1]
    dense = np.zeros((nn,))

    ############# # 计算到虚拟点的距离和维度和 # ###############
    distance = np.sqrt(np.sum(np.square(data - point), axis=1))
    dis_index = np.argsort(distance)  # 排序的索引
    distance = np.sort(distance)# 从小到大排序

    max_dis = np.max(distance)
    min_dis = np.min(distance)
    tt = int(np.ceil((max_dis - min_dis) / r))
    dim_sum = np.zeros((nn, ))
    for i in range(nn):
        dim_sum[i] = np.sum(data[dis_index[i]])

    ############# # 划分公共圆环，即记录每个公共圆环的第一个点的索引，若该公共圆环为空，则索引为-1 # #############
    flag = 0
    length = 0
    annular_gap = r
    annular_head = [[-1, 0] for i in range(tt)]
    annular_head[0][0] = 0
    for i in range(nn):
        while (distance[i]) - min_dis > annular_gap:
            annular_gap += r
            annular_head[flag + 1][0] = i
            annular_head[flag][1] = length
            flag += 1
            length = 0
        length += 1
    annular_head[flag][1] = length

    ########## # 划分为公共圆环 # #############
    # annular = [[] for i in range(tt)]# 圆环
    # annular_gap = r# 圆环的外径
    # count = 0# 第几个圆环
    # for i in range(nn):
    #     while (distance[i]) - min_dis > annular_gap:
    #         annular_gap += r
    #         count += 1
    #     if((distance[i]) - min_dis <= annular_gap):
    #         annular[count].append(i)

    ######## # 密度计算，对于第i个公共圆中的所有点，只需考虑i-1、i、i+1个公共圆环，但考虑边界条件 # ##########
    # position_in_annular = 0
    for i in range(tt):
        if annular_head[i][1] == 0:
            continue
        if i == 0:
            head = 0
            tail = annular_head[0][1] + annular_head[1][1]
            start = 0# 在i-1,i,i+1公共圆环中，第i个公共圆环的起始位置
        elif i == tt - 1:
            head = nn - annular_head[-1][1] - annular_head[-2][1]
            tail = -1
            start = annular_head[-2][1]
        elif i == tt - 2:
            head = nn - annular_head[-1][1] - annular_head[-2][1] - annular_head[-3][1]
            tail = -1
            start = annular_head[-3][1]
        else:
            head = annular_head[i][0] - annular_head[i-1][1]
            tail = annular_head[i][0] + annular_head[i][1] + annular_head[i+1][1]
            start = annular_head[i-1][1]

        ########## # 对三个公共圆环按照维度和进行排序 # ###########
        if tail == -1:
            triple_dim = dim_sum[head:]
        else:
            triple_dim = dim_sum[head:tail]
        triple_dim_sort = np.sort(triple_dim)# 我们需要知道第i个圆环中所有的点排序后在annular_dim_sort中的位置
        triple_dim_sort_idx = np.argsort(triple_dim)# 三个圆环内数据点按照维度和排序后索引
        triple_dim_map = [0 for t in range(triple_dim.shape[0])]# 三个圆环中每个点按照维度和排序后 在triple_dim_sort中的位置
        for j in range(triple_dim_sort.shape[0]):
            triple_dim_map[triple_dim_sort_idx[j]] = j

        # 此处有问题。需改
        end = triple_dim_sort.shape[0] - 1
        if i == 0:
            bias = 0
        else:
            bias = annular_head[i - 1][1]# 如果直接使用annular_head[i-1][0]定位三个圆环的起点，当i为0时无效，所以使用annular_head[i][0] - bias来定义三个圆环的起点，只需在循环前计算bias
        for j in range(int(annular_head[i][1])):
            pre = triple_dim_map[start + j] - 1
            lat = triple_dim_map[start + j] + 1
            pst_in_data = dis_index[annular_head[i][0] + j]# 在 data 中的索引
            while pre >= 0 and np.abs(triple_dim_sort[pre] - triple_dim[start + j]) <= np.sqrt(dd) * r:# triple_dim[start + j] == triple_dim_sort[triple_dim_map[start+j]]
                pre_in_data = dis_index[annular_head[i][0] - bias + triple_dim_sort_idx[pre]]# triple_dim_sort_idx[pre]对应排序前的索引，pre对应排序后的索引
                if np.sqrt(np.sum(np.square(data[pst_in_data] - data[pre_in_data]))) <= r:
                    dense[pst_in_data] += 1
                pre -= 1
            while lat <= end and np.abs(triple_dim_sort[lat] - triple_dim[start + j]) <= np.sqrt(dd) * r:
                lat_in_data = dis_index[annular_head[i][0] - bias + triple_dim_sort_idx[lat]]  # triple_dim_sort_idx[lat]对应排序前的索引，lat对应排序后的索引
                if np.sqrt(np.sum(np.square(data[pst_in_data] - data[lat_in_data]))) <= r:
                    dense[pst_in_data] += 1
                lat += 1
    return dense
