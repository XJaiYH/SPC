import numpy as np
from scipy import io
import pandas as pd
data = io.loadmat("../../data/Scene15.mat")
print(data)
label = data['Y']
data = data['X']
data = np.array(data)
print(data)
print(np.sum(np.isnan(data[0])))