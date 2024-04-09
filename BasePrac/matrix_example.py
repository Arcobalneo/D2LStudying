import torch

# 标量
x = torch.tensor(3.5)

# 向量
y = torch.tensor([1.0, 2.0, 3.0])
z = torch.ones(3, dtype=torch.float32)
print('向量点积')
print(torch.dot(y, z))

# 矩阵
z = torch.tensor(
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ])
print(z.T)  # 转置

# 张量
p = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print('原始矩阵')
print(p)

print('逐元素乘法')
q = torch.ones((3, 4)) + 1
print(p * q)

print('矩阵向量积')
a = torch.ones(4, dtype=torch.float32) + 2
print(torch.mv(p, a))

print('矩阵乘法')
b = torch.ones([4, 3], dtype=torch.float32) + 1
print(torch.mm(p, b))

# 向量 L1范数 = 所有分量绝对值之和
print('L1范数')
u = torch.zeros(3) - 2.0
print(torch.abs(u).sum())

# 向量 L2范数 = 所有分量平方之和，再开根号
print('L2范数')
u = torch.tensor([3.0, 4.0])
print(u.norm())
m = torch.ones(9, dtype=torch.float32).reshape(3, 3)
print(torch.norm(m))  # 矩阵L2范数也是所有元素平方和开根
