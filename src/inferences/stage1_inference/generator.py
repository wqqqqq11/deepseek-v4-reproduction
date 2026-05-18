"""
文本生成器模块

提供流式生成功能，支持从检查点加载模型。
"""

import torch
from pathlib import Path
from typing import Iterator, Optional

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.models.config import ModelArgs
from src.models.transformer_stage1 import TransformerStage1
from .sampler import sample_next_token


class Generator:
    """
    文本生成器。

    支持流式生成，可逐个token输出。

    Args:
        model: 模型实例。
        tokenizer: tokenizer实例。
        device: 计算设备。
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: dict):
        """
        从检查点创建生成器。

        Args:
            checkpoint_path: 检查点文件路径。
            config: 配置字典（从yaml加载）。

        Returns:
            Generator: 配置好的生成器实例。
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers库未安装，无法加载tokenizer")

        device = torch.device(config["hardware"]["device"])

        # 加载tokenizer
        tokenizer_name = config["data"].get("tokenizer_name", "jingyaogong/minimind-3")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        # 模型配置
        model_args = ModelArgs.stage1()
        model_args.vocab_size = config["model"]["vocab_size"]
        model_args.dim = config["model"]["dim"]
        model_args.n_layers = config["model"]["n_layers"]
        model_args.n_heads = config["model"]["n_heads"]
        model_args.max_seq_len = config["model"]["max_seq_len"]
        model_args.q_lora_rank = config["model"]["q_lora_rank"]
        model_args.kv_lora_rank = config["model"]["kv_lora_rank"]
        model_args.head_dim = config["model"]["head_dim"]

        # 加载模型
        model = TransformerStage1(model_args).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])

        return cls(model, tokenizer, device)

    def encode(self, text: str) -> list:
        """将文本编码为token IDs。"""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list) -> str:
        """将token IDs解码为文本。"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        生成文本（非流式）。

        Args:
            prompt: 输入提示文本。
            max_new_tokens: 最大生成token数。
            strategy: 采样策略。
            temperature: 温度参数。
            top_k: Top-K参数。
            top_p: Top-P参数。

        Returns:
            str: 生成的完整文本。
        """
        tokens = []
        for token_id in self.generate_stream(
            prompt, max_new_tokens, strategy, temperature, top_k, top_p
        ):
            tokens.append(token_id)

        return self.decode(tokens)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[int]:
        """
        流式生成token。

        Args:
            prompt: 输入提示文本。
            max_new_tokens: 最大生成token数。
            strategy: 采样策略。
            temperature: 温度参数。
            top_k: Top-K参数。
            top_p: Top-P参数。

        Yields:
            int: 生成的token ID。
        """
        input_ids = self.encode(prompt)
        generated = list(input_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([generated], dtype=torch.long, device=self.device)
                logits = self.model(x)

                next_logits = logits[0, -1, :]
                next_token = sample_next_token(
                    next_logits, strategy, temperature, top_k, top_p
                )

                generated.append(next_token)
                yield next_token
