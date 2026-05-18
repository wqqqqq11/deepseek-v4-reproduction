"""
模型前向传播测试

验证模型架构正确性，无需加载检查点。
"""

import torch
from src.models.config import ModelArgs
from src.models.transformer_stage1 import TransformerStage1


def test_forward_shape():
    """测试输入输出形状。"""
    args = ModelArgs.stage1()
    model = TransformerStage1(args)

    batch_size, seq_len = 2, 128
    x = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    logits = model(x)

    expected_shape = (batch_size, seq_len, args.vocab_size)
    assert logits.shape == expected_shape, f"形状不匹配: {logits.shape} != {expected_shape}"

    return True


def test_gradient_flow():
    """测试梯度能正常回传。"""
    args = ModelArgs.stage1()
    model = TransformerStage1(args)

    x = torch.randint(0, args.vocab_size, (2, 64))
    logits = model(x)

    # 构造虚拟损失并反向传播
    loss = logits.sum()
    loss.backward()

    # 检查至少有一些梯度不为零
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "梯度未正确回传"

    return True


def test_numerical_stability():
    """测试数值稳定性。"""
    args = ModelArgs.stage1()
    model = TransformerStage1(args)

    # 多次前向传播检查数值稳定性
    for _ in range(10):
        x = torch.randint(0, args.vocab_size, (2, 128))
        logits = model(x)

        assert not torch.isnan(logits).any(), "输出包含NaN"
        assert not torch.isinf(logits).any(), "输出包含Inf"

    return True


def run_tests():
    """运行所有前向测试。"""
    print("  测试输入输出形状...")
    test_forward_shape()
    print("  ✓ 形状测试通过")

    print("  测试梯度回传...")
    test_gradient_flow()
    print("  ✓ 梯度测试通过")

    print("  测试数值稳定性...")
    test_numerical_stability()
    print("  ✓ 数值稳定性测试通过")

    return True
