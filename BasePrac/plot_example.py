import matplotlib.pyplot as plt
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决torch重复初始化dll warning，影响未知

x = torch.arange(0, 3, 0.1)
y = torch.sin(x)
print(x)
print(y)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y, "b--", label="sin")
plt.legend(loc='best')
plt.show()
