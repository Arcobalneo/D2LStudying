import torch
import matplotlib.pyplot as plt
from torch.distributions import multinomial

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决torch重复初始化dll warning，影响未知

fair_prob = torch.ones([6]) / 6  # [1/6,1/6,1/6,……]
sample = multinomial.Multinomial(1000, fair_prob).sample()  # 该概率分布下1000次采样结果
print(sample/1000)

counts = multinomial.Multinomial(10, fair_prob).sample((500,))  # 500组，每组抽10次
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)


for i in range(6):
    plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
plt.axhline(y=0.167, color='black', linestyle='dashed')  # 结果逼近于理论概率1/6
plt.legend()
plt.show()