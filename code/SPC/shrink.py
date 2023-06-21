import numpy as np
from scipy.spatial import KDTree


def shrink(data: np.ndarray, density: np.ndarray, neighbor: int, T=1) -> np.ndarray:
    '''

    Parameters
    ----------
    data: there has not any 'nan' in data that as input of this step
    neighbor: k-nearest-neighbors
    T:
    Returns:dataset after ameliorating
    -------

    '''
    bata = data.copy()
    pointNum = data.shape[0]
    tree = KDTree(bata)
    distance_sort, neighbor_idx = tree.query(bata, neighbor + 1)
    density_mean = np.mean(density)
    G = np.sum(distance_sort[:, 1]) / data.shape[0]
    for i in range(data.shape[0]):
        if np.isnan(data[i]).any() == True:
            continue
        if density[i] >= density_mean:
            continue
        displacement = 0.
        for j in range(1, neighbor + 1):
            ff = (data[neighbor_idx[i][j]] - data[i])
            fff = (distance_sort[i, 1] / (
                    distance_sort[i, j] * distance_sort[i, j]))
            displacement += G * ff * fff
        bata[i] = data[i] + displacement * T  # object after moving
    return bata
