import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def computeVar(data: np.ndarray) -> list:
    nn = data.shape[0]
    dd = data.shape[1]
    res = [0 for i in range(dd)]
    for i in range(dd):
        var = np.nanvar(data[:, i])
        res[i] = [var]
    return res

def getDenseGraph(project: np.ndarray, r: float) -> np.ndarray:
    nn = project.shape[0]
    dense_graph = np.zeros((nn, ))
    project_sorted = np.sort(project)
    print(project_sorted)
    head = 0
    tail = 1
    tail_old = 0
    count = 0
    flag = False
    while tail < nn:
        if head == tail:
            tail_old = tail
            count = 0
            tail += 1
        elif project_sorted[tail] - project_sorted[head] <= r:
            count += 1
            tail += 1
            flag = True
        else:
            if flag == True:
                dense_graph[head:tail] += count
                if tail_old > head:
                    dense_graph[tail_old:tail] += tail_old - head - 1
                tail_old = tail
                flag = False
            head += 1
            count = 0
            if np.isnan(project_sorted[tail]) == True:
                break
    if flag == True:
        dense_graph[head:tail] += count
        dense_graph[tail_old:tail] += tail_old - head - 1

    dense_2 = np.zeros((nn, ))
    for i in range(nn):
        for j in range(nn):
            if i != j and abs(project_sorted[i] - project_sorted[j]) <= r:
                dense_2[i] += 1
    print(np.sum(dense_graph == dense_2))
    print(dense_graph)
    print(dense_2)
    return dense_graph, project_sorted

def plotDenseGraph(dense_graph: np.ndarray, sorted_project: np.ndarray) -> list:
    x = sorted_project.reshape((-1, ))
    fig, ax = plt.subplots()
    ax.plot(x, dense_graph)
    ax.set_title("density decision")
    print("请点击坐标点，按任意键退出...")
    points = plt.ginput(n=-1)
    plt.show()
    border = []
    for point in points:
        border.append(point[0])
    return border

def getProjectCluster(project: np.ndarray, border: list,  project_cluster: np.ndarray):
    nn = project.shape[0]
    border = sorted(border)
    for i in range(nn):
        if np.isnan(project[i]) == True:
            continue
        for j in range(len(border)):
            if project[i] <= border[j]:
                project_cluster[i] = j
                break
            else:
                project_cluster[i] = j+1

def project_clustering(data: np.ndarray, pp: int):
    dd = data.shape[1]
    nn = data.shape[0]
    vars = computeVar(data)
    vars = np.array(vars).reshape((-1, ))
    dims = np.argsort(-vars)
    project_cluster_pre = -np.ones((nn, ), dtype=int)

    for i, dim in enumerate(dims):
        project = data[:, dim].reshape(-1,)
        dense_graph, sorted_project = getDenseGraph(project, 0.05)
        border = plotDenseGraph(dense_graph, sorted_project)
        print(border)
        if i == 0:
            getProjectCluster(project, border, project_cluster_pre)
            continue
        else:
            project_cluster_lat = -np.ones((nn,), dtype=int)
            getProjectCluster(project, border, project_cluster_lat)
            cluster_map = {}# 簇的映射关系
            cluster_pre = {}# 第i次的投影簇，只有两维度上都存在的点
            cluster_lat = {}# 第i+1次的投影簇，只有两维度上都存在的点
            incomplete_pre = {}# 第i次的投影簇，只有第i维存在，第i+1维不存在的点
            incomplete_lat = {}# 第i次的投影簇，只有第i+1维存在，第i维不存在的点
            count = 0# 融合之后的簇的编号

            ###### # 在两个维度上不缺失点的聚类 # ########
            for j, item in enumerate(zip(project_cluster_pre, project_cluster_lat)):
                if item[0] != -1 and item[1] != -1:
                    key = str(item[0]) + str(item[1])
                    if cluster_map.get(key) is None:
                        cluster_map[key] = count
                        count += 1
                    project_cluster_pre[j] = cluster_map[key]
                    if cluster_pre.get(item[0]) is None:
                        cluster_pre[item[0]] = []
                    if cluster_lat.get(item[1]) is None:
                        cluster_lat[item[1]] = []
                    cluster_pre[item[0]].append(j)
                    cluster_lat[item[1]].append(j)
                elif item[0] == -1 and item[1] != -1:
                    if incomplete_lat.get(item[1]) is None:
                        incomplete_lat[item[1]] = []
                    incomplete_lat[item[1]].append(j)
                elif item[0] != -1 and item[1] == -1:
                    if incomplete_pre.get(item[0]) is None:
                        incomplete_pre[item[0]] = []
                    incomplete_pre[item[0]].append(j)

            ###### # 在两个维度上其中一个维度存在缺失的点的聚类 # ########
            for key, val in incomplete_pre.items():
                num_dict = {}
                cluster = cluster_pre[key]
                for item in cluster:
                    cluster_number = project_cluster_pre[item]
                    num_dict[cluster_number] = num_dict.get(cluster_number, 0) + 1
                max_num = 0
                for kkey, vval in num_dict.items():
                    if vval > max_num:
                        max_num = vval
                        cls = kkey
                project_cluster_pre[val] = cls

            for key, val in incomplete_lat.items():
                num_dict = {}
                cluster = cluster_lat[key]
                for item in cluster:
                    cluster_number = project_cluster_pre[item]
                    num_dict[cluster_number] = num_dict.get(cluster_number, 0) + 1
                max_num = 0
                for kkey, vval in num_dict.items():
                    if vval > max_num:
                        max_num = vval
                        cls = kkey
                project_cluster_pre[val] = cls
            print(cluster_map)
        if len(cluster_map) >= pp:
            break
    return project_cluster_pre

def plotCluster(data: np.ndarray, labels: np.ndarray, title: str, save_path: str):
    fig, ax = plt.subplots()
    sortNumMax = np.max(labels)
    sortNumMin = np.min(labels)
    color = [
        "#900C3F",  # 紫红色
        "#006400",  # 深绿色
        "#4B0082",  # 靛青色
        "#FF4500",  # 橙红色
        "#FF1493",  # 深粉色
        "#008B8B",  # 深青色
        "#FF7F50",  # 珊瑚色
        "#4682B4",  # 钢蓝色
        "#A9A9A9",  # 暗灰色
        "#556B2F",  # 暗绿色
        "#9370DB",  # 中紫色
        "#8B7355",  # 赭色
        "#FFD700",  # 库金色
        "#2E8B57",  # 海洋绿色
        "#008B8B",  # 暗藏青色
        "#BDB76B",  # 黄褐色
        "#654321",  # 深棕色
        "#9400D3",  # 暗紫色
        "#008080",  # 暗青色
        "#CD5C5C"  # 褐红色
    ]
    # color = ['#125B50', '#4D96FF', '#FFD93D', '#FF6363', '#CE49BF', '#22577E', '#4700D8', '#F900DF', '#95CD41',
    #          '#FF5F00', '#40DFEF', '#8E3200', '#001E6C', '#C36A2D', '#B91646']
    lineform = ['o']
    for i in range(sortNumMin, sortNumMax + 1):
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        fontSize = 15
        colorNum = (i - sortNumMin) % len(color)
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(title, fontsize=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
