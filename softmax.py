import torch
# 导入库
import torch.nn as nn
import math
scores=torch.ones(3, 4)
attention_weights = torch.softmax(scores, dim=-1)
print("attention_weights:", attention_weights)