import numpy as np
import pandas as pd

file_path = "../../data/dim32.txt"
save_path = "../../data/dim32-incomplete.csv"
loss_rate = 0.1

data = np.loadtxt(file_path)
columns = []
label = data[:, -1].reshape((-1, 1))
data = data[:, :-1]
for i in range(data.shape[1]):
    columns.append("dim"+str(i))
columns.append("label")
df = pd.DataFrame(np.concatenate((data, label), axis=1), columns=columns)

### 缺失
# loss_num = int(np.ceil(data.shape[0] * loss_rate))
# idx = np.random.choice(np.arange(0, data.shape[0]), size=loss_num, replace=False)
# idx_pre = idx[:loss_num // 2]
# idx_lat = idx[loss_num // 2:]
# df.iloc[idx_pre, 0] = np.nan
# df.iloc[idx_lat, 1] = np.nan
# df.iloc[0, 0] = np.nan
# df.iloc[0, 1] = np.nan

df.to_csv(save_path, index=False)