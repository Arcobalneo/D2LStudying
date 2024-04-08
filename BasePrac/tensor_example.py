import torch

# region tensor init
x = torch.arange(12)
print(x)
x = x.reshape(3, -1)  # 用-1表示，在给出高度情况下，函数会自动计算出列数
print(x)
print(x.shape)

y = torch.ones((2, 2, 3))  # 零矩阵则用zeros
print(y)

z = torch.randn(3, 4)  # 标准正态分布初始化
print(z)

p = torch.tensor([[1, 2, 3], [3, 2, 1]])  # 手动初始化
print(p)
print(p.shape)
# endregion

# region tensor op between same shape
x = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)
y = torch.tensor(
    [
        [-1, -2, -3],
        [-4, -5, -6]
    ]
)

# 逐元素的运算
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)  # **是求幂次
print(torch.exp(x))  # e**x运算

# 张量连结
print(torch.cat((x, y), dim=0))
print(torch.cat((x, y), dim=1))

# 逻辑运算
print(x == y)
print(x > y)

# 所有元素求和
print(torch.sum(x))
# endregion

# region tensor op between different shape
a = torch.arange(3).reshape((3, 1))
b = torch.tensor([0,-1,-2])
print(a+b)  # 把3X1和1X2矩阵，相加得到3X2，相加之前会先各自copy行或列拓展到3X2
# endregion

# region tensor index
x = torch.arange(12).reshape((3,4))
print(x[0])
print(x[-1])
print(x[1:3])  # 切片是左闭右开的
x[1:3,:] = 999  # 单一个:符号，表示全选
print(x)
# endregion

# region 节省内存
a = torch.arange(12).reshape((3,4))
b = torch.zeros_like(a)
before = id(b)  # 查地址
b = a + b  # 会导致b的地址改变，产生一次新的内存分配
print(id(b) == before)

# 原地更新优化写法1
before = id(b)
b += a
print(id(b) == before)

# 原地更新优化写法2，用切片语法
before = id(b)
b[:] = b + a
print(id(b) == before)
# endregion

# region 与numpy相互转换
a = a.numpy()
print(type(a))
b = torch.tensor(a)
print(type(b))

# 大小为1 转 单个标量
t = torch.tensor([3.99])
print(t)
print(t.item())
print(float(t))
# endregion
