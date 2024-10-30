# A矩阵 （100，512）乘以 B矩阵 （512，100）得到（100，100）
# 将A矩阵 （100，512）分成多头，假设为8，（8，100，64）乘以 B矩阵 （8，64，100）得到（8，100，100），最后将（8，100，100）wise加和得到（100，100）两种方式得到的结果一样吗
import torch

# 设置随机种子以确保每次运行结果一致
torch.manual_seed(0)
# A is Query, B is Key
# 定义 A 和 B 矩阵
A = torch.randn(100, 512)
B = torch.randn(512, 100)

# 直接计算 A * B
AB_direct = A @ B

# 将 A 和 B 分成 8 头
num_heads = 8
A_heads = A.view(100, num_heads, -1).permute(1, 0, 2)  # 转换为 (8, 100, 64)
B_heads = B.view(num_heads, -1, 100)  # 转换为 (8, 64, 100)

# 计算每个头的结果并加和
AB_heads = torch.bmm(A_heads, B_heads).sum(dim=0)  # (8, 100, 100) -> (100, 100)

# 比较两种方法的结果，容差设置为 0.01
are_equal = torch.allclose(AB_direct, AB_heads, atol=1e-2)

print(f"Are the results from both methods equal within a tolerance of 0.01? {are_equal}")
print("Direct multiplication result:")
print(AB_direct)
print("Multi-head sum result:")
print(AB_heads)


''' 
              [ b1
                b2    = a1b1+a2b2+a3b3+a4b4   X（t1）相对于X（t2）的attention score
[a1 a2 a3 a4]*  b3
                b4]


[a1 a2] * [b1 
            b2]= a1b1+a2b2                X（t1）前半部分 相对于X（t2）前半部分的attention score

[a3 a4]   [b3
            b4] = a3b3+a4b4                X（t1）后半部分 相对于X（t2）后半部分的attention score

obviously, a1b1+a2b2+a3b3+a4b4  =  a1b1+a2b2     +    a3b3+a4b4 
也即8个score的（100，100） elem wise add =单头的score（100，100） 单头的一个值由两个512向量相乘得到，8个头由两个512向量切分为8个64向量，然后对应下标的两个64向量相乘得到，原本8个（100，100）elem wise add等于单头的（100，100）但是 attention_weights = torch.softmax(scores, dim=-1)这一个softmax的非线性打破了这个等式
另，这里的（100，100）就是邻接矩阵的意思，也即一个节点对其他节点的attention score，聚类也经常用到（均值漂移聚类）。

也可见mha和sha的网络参数量是一样的。TODO 计算复杂度？

总结：mha的核心是将一个512维的向量切分为8个64维的向量，对应下标的 64维向量做自己独立的attention score矩阵。可以认为，MHA适用的条件是特征向量的 维度过高以至于有部分维度可以组成独立的语义，如512维度向量前半256维度表示一个属性，后面表示另外属性。

'''

