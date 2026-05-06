"""
检查八层的小deepseek的梯度是否正常流转
"""


import torch
from src.models.config import ModelArgs
from src.models.transformer import Transformer

args = ModelArgs.tiny()                    
model = Transformer(args)
tokens = torch.randint(0, args.vocab_size, (2, 128))
logits = model(tokens)                     # 不需要 start_pos（训练模式）

loss = logits.mean()                       # 简单标量 loss
loss.backward()                            # 应该不报错

# 检查每层都有梯度
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"[X] {name}: grad is None")
    else:
        print(f"[OK] {name}: grad shape {param.grad.shape}")