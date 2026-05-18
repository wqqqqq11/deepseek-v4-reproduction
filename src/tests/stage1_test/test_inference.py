"""
推理功能测试

验证模型生成能力，需要加载训练好的检查点。
"""

import torch


def generate_simple(model, prompt_tokens, max_new_tokens, temperature=1.0):
    """
    简单的自回归生成函数。

    Args:
        model: 模型实例。
        prompt_tokens: 提示token列表。
        max_new_tokens: 最大生成token数。
        temperature: 采样温度。

    Returns:
        list: 生成的token列表。
    """
    model.eval()
    device = next(model.parameters()).device
    generated = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(x)

            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

    return generated


def test_generation_loop(model, vocab_size):
    """测试生成循环能正常跑通。"""
    prompt = [1, 2, 3]  # 虚拟prompt

    output = generate_simple(model, prompt, max_new_tokens=10)

    assert len(output) == 13, f"生成长度错误: {len(output)}"
    assert all(0 <= t < vocab_size for t in output), "生成非法token"

    return True


def test_generation_samples(model, tokenizer=None):
    """测试固定prompts的生成效果。"""
    prompts = [
        [1, 2],      # 你好（示例）
        [3, 4, 5],   # The meaning...
        [6, 7],      # 1+1=
        [8, 9, 10],  # def hello_world
    ]

    results = []
    for i, prompt in enumerate(prompts):
        output = generate_simple(model, prompt, max_new_tokens=20)
        results.append({
            'prompt_id': i,
            'output_length': len(output),
            'output_tokens': output
        })

    return results


def run_tests(model, vocab_size, tokenizer=None):
    """运行所有推理测试。"""
    print("  测试生成循环...")
    test_generation_loop(model, vocab_size)
    print("  ✓ 生成循环测试通过")

    print("  测试样例生成...")
    samples = test_generation_samples(model, tokenizer)
    print(f"  ✓ 生成{len(samples)}个样例")

    for s in samples:
        print(f"    Prompt {s['prompt_id']}: {s['output_length']} tokens")

    return True
