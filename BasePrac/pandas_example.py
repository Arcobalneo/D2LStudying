import os
import numpy as np
import pandas as pd
import torch
# region create dataset manually
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'pandas_example_data.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('6,NA,140000\n')
    f.write('NA,Peter,160000\n')  # 每行表示一个数据样本
# endregion

# region read dataset
dataset = pd.read_csv(data_file)
print(dataset)
# endregion

# region fill NaN

inputs = dataset.iloc[:, 0:2]
print(inputs)

# 处理离散值
# 由于“Alley”列列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”
inputs = pd.get_dummies(inputs)

# 数值类型的缺失 均值法
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 删除法 删除缺省值最多的列
# 计算每列的缺失值数量
null_counts = dataset.isnull().sum()

# 找到缺失值最多的列
most_missing_column = null_counts.idxmax()

# 删除该列
data = dataset.drop(most_missing_column, axis=1)
print(data)
# endregion

# region inputs to tensor
x = torch.tensor(inputs.to_numpy(dtype=np.float32))  # 性能考虑，一般用float32，而不用默认float的float64
print(x)
# endregion