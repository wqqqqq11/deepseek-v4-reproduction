"""
训练配置模块

定义阶段1预训练的所有超参配置，使用 dataclass 实现类型安全和便捷实例化。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class OptimizerConfig:
    """优化器配置"""
    # Muon 参数组（使用 Muon 优化的模块名称列表）
    muon_groups: List[str] = field(default_factory=lambda: ["attn", "mlp"])
    # AdamW 参数组（使用 AdamW 优化的模块名称列表）
    adamw_groups: List[str] = field(default_factory=lambda: ["embed", "head", "norm", "bias"])
    
    # 学习率
    muon_lr: float = 3e-4
    adamw_lr: float = 3e-4
    
    # Muon 固定超参（论文 Algorithm 1）
    muon_momentum: float = 0.95
    muon_gamma: float = 0.18
    muon_nesterov: bool = True
    
    # AdamW 参数
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    
    # 权重衰减
    weight_decay: float = 0.1


@dataclass
class LRSchedulerConfig:
    """学习率调度配置（三段式）"""
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    
    # 各阶段步数（基于 20K steps/epoch，2 epochs = 40K total）
    warmup_steps: int = 2000
    peak_steps: int = 6000
    cosine_steps: int = 12000
    
    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.peak_steps + self.cosine_steps


@dataclass
class MTPConfig:
    """多令牌预测配置"""
    num_future_tokens: int = 2
    loss_weight: float = 0.3
    loss_weight_decay_end: float = 0.1


@dataclass
class PrecisionConfig:
    """精度配置"""
    fp8_enabled: bool = True
    compute_dtype: str = "bfloat16"
    optimizer_state_dtype: str = "float16"


@dataclass
class DataConfig:
    """数据配置"""
    train_bin: str = "datasets/stage_1_datasets/6_binarize/train.bin"
    val_bin: str = "datasets/stage_1_datasets/6_binarize/val.bin"
    context_size: int = 1024
    
    # 采样参数
    shuffle_seed: int = 42


@dataclass
class LoggingConfig:
    """日志配置"""
    log_dir: str = "logs/stage1"
    checkpoint_dir: str = "checkpoints/stage1"
    
    save_best_only: bool = True
    eval_every_epoch: bool = True
    
    log_every_steps: int = 10
    tensorboard_port: int = 6006


@dataclass
class HardwareConfig:
    """硬件配置"""
    device: str = "cuda"
    num_gpus: int = 1
    use_compile: bool = False


@dataclass
class TrainConfig:
    """
    阶段1预训练完整配置
    
    聚合所有子配置，提供统一访问接口。
    """
    
    # 训练流程参数
    num_epochs: int = 2
    batch_size: int = 32
    gradient_accumulation: int = 1
    grad_clip: float = 1.0
    
    # 子配置
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # 随机种子
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件"""
        import yaml
        
        def convert(obj):
            if isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            if hasattr(obj, "__dataclass_fields__"):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            return obj
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(convert(self), f, default_flow_style=False, allow_unicode=True)


def get_stage1_config() -> TrainConfig:
    """获取阶段1默认配置"""
    return TrainConfig()
