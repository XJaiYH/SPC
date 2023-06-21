import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from getDensity import compute_dense
from shrink import shrink
from clustering import project_clustering, plotCluster

if __name__=="__main__":
    file_path = "../../data/Aggregation-incomplete.csv"
    file_type = file_path.split(".")[-1]
    pca = PCA(n_components=2)
    pp = 7# 最大预估簇数
    dd = 1
    ########## # load data # ############
    if file_type == "csv":
        data = pd.read_csv(file_path)
        label = data['label'].values.astype(int)
        data = data.iloc[:, :-1].values
    elif file_type == "txt":
        data = np.loadtxt(file_path)
        label = data[:, -1].astype(int)
        data = data[:, :-1]

    ####################### # normalization # ##################################
    for j in range(data.shape[1]):
        max_ = np.nanmax(data[:, j])
        min_ = np.nanmin(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            if np.isnan(data[i][j]) == False:
                data[i][j] = (data[i][j] - min_) / (max_ - min_)

    ####### # 分离缺失数据的数据点和不缺失数据的数据点，以求密度并信息增强 # #########
    incomplete_idx = []
    for i in range(data.shape[0]):
        if np.isnan(data[i]).any() == True:
            incomplete_idx.append(i)

    incomplete_data = data[incomplete_idx]
    incomplete_label = label[incomplete_idx]
    data = np.delete(data, incomplete_idx, axis=0)
    label = np.delete(label, incomplete_idx)

    ######### # r # #########
    tree = KDTree(data)
    dis, dis_idx = tree.query(data, k=int(np.ceil(0.02 * data.shape[0])) - 1)  # round(data.shape[0] * 0.02)-1)
    r = np.mean(dis[:, -1])
    print("r: ", r)

    ###### # 虚拟点 # #######
    point = []
    for i in range(data.shape[1]):
        point.append(-9999)
    point = np.array(point).reshape((-1,))

    dense = compute_dense(data, point, r)  # compute density
    for i in range(dd):
        data = shrink(data, dense, 5, 0.5)  # information augment
        # data = bata
    plotCluster(data, label, "after shrink", "../../result/shrink.png")

    ########## # 合并不缺失数据的点和缺失数据的点，进行单维聚类 # #############
    data = np.concatenate((data, incomplete_data), axis=0)
    label = np.concatenate((label, incomplete_label), axis=0).reshape((-1,))
    cluster = project_clustering(data, pp)  # 进行单维聚类

    ####### # to visualize# ######
    data = np.nan_to_num(data)
    if data.shape[1] > 2:
        data = pca.fit_transform(data)
    plotCluster(data, cluster, "result", "../../result/compare.png")
