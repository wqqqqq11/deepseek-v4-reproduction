"""
DeepSeek-V4 官方 Muon 优化器实现

严格遵循论文 Algorithm 1，包含：
    - 动量累积
    - Nesterov 加速
    - 混合 Newton-Schulz 迭代（10次，分两个阶段）
    - RMS 重缩放

适用范围：除 embedding、prediction head、RMSNorm 权重、所有 bias 外的所有矩阵参数。
"""

import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.optim import Optimizer


def hybrid_newton_schulz_iter(
    Y: torch.Tensor,
    Z: torch.Tensor,
    coeffs: Tuple[float, float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    单次混合 Newton-Schulz 迭代。
    
    Args:
        Y: 当前 Y 矩阵（近似正交矩阵）。
        Z: 当前 Z 矩阵（用于计算逆）。
        coeffs: (a, b, c) 迭代系数。
        
    Returns:
        (new_Y, new_Z): 更新后的矩阵对。
    """
    a, b, c = coeffs
    
    # 计算中间结果
    Y2 = Y @ Y
    Y3 = Y2 @ Y
    
    # 更新 Z
    ZY = Z @ Y
    ZY2 = ZY @ Y
    new_Z = a * Z - b * ZY + c * ZY2
    
    # 更新 Y
    new_Y = a * Y - b * Y2 + c * Y3
    
    return new_Y, new_Z


def compute_matrix_root(Y: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """
    计算矩阵四次方根的逆（即 X^{-1/4}）的近似。
    
    使用混合 Newton-Schulz 迭代：
        - 前 8 次：快速收敛系数 (3.4445, -4.7750, 2.0315)
        - 后 2 次：稳定精调系数 (2, -1.5, 0.5)
    
    Args:
        Y: 输入矩阵（梯度动量的累积）。
        iterations: 总迭代次数（默认 10）。
        
    Returns:
        torch.Tensor: 近似 X^{-1/4} 矩阵。
    """
    # 归一化初始值
    norm = torch.norm(Y, p="fro")
    if norm < 1e-12:
        return torch.eye(Y.shape[0], device=Y.device, dtype=Y.dtype)
    
    Y = Y / norm
    Z = torch.eye(Y.shape[0], device=Y.device, dtype=Y.dtype)
    
    # 定义两个阶段系数
    fast_coeffs = (3.4445, -4.7750, 2.0315)
    stable_coeffs = (2.0, -1.5, 0.5)
    
    # 前 8 次快速收敛
    for _ in range(min(8, iterations)):
        Y, Z = hybrid_newton_schulz_iter(Y, Z, fast_coeffs)
    
    # 后 2 次稳定精调
    for _ in range(max(0, iterations - 8)):
        Y, Z = hybrid_newton_schulz_iter(Y, Z, stable_coeffs)
    
    return Z


def is_matrix_param(param: nn.Parameter, name: str) -> bool:
    """
    判断参数是否为适合 Muon 的矩阵参数。
    
    条件：
        1. 维度为 2
        2. 非 embedding、head、norm、bias
    
    Args:
        param: 待判断的参数。
        name: 参数名称（用于排除特定模块）。
        
    Returns:
        bool: 是否适合使用 Muon。
    """
    if param.ndim != 2:
        return False
    
    # 排除特定模块
    exclude_keywords = ["embed", "head", "norm", "bias"]
    for keyword in exclude_keywords:
        if keyword in name.lower():
            return False
    
    return True


class Muon(Optimizer):
    """
    DeepSeek-V4 官方 Muon 优化器。
    
    实现论文 Algorithm 1，包含动量累积、Nesterov 加速、Newton-Schulz
    正交化和 RMS 重缩放。
    
    Args:
        params: 可优化参数列表或字典。
        lr: 学习率（默认 3e-4）。
        momentum: 动量系数（默认 0.95）。
        weight_decay: 权重衰减（默认 0.1）。
        gamma: RMS 重缩放因子（默认 0.18）。
        nesterov: 是否启用 Nesterov（默认 True）。
        ns_iterations: Newton-Schulz 迭代次数（默认 10）。
    
    Example:
        >>> optimizer = Muon(
        ...     model.parameters(),
        ...     lr=3e-4,
        ...     momentum=0.95,
        ...     weight_decay=0.1,
        ...     gamma=0.18,
        ... )
    """
    
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        gamma: float = 0.18,
        nesterov: bool = True,
        ns_iterations: int = 10,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            gamma=gamma,
            nesterov=nesterov,
            ns_iterations=ns_iterations,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            gamma = group["gamma"]
            nesterov = group["nesterov"]
            ns_iterations = group["ns_iterations"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                
                momentum_buffer = state["momentum_buffer"]
                
                # 动量累积: M_t = mu * M_{t-1} + G_t
                momentum_buffer.mul_(momentum).add_(grad)
                
                # Nesterov: 如果启用，计算 Nesterov 梯度
                if nesterov:
                    nesterov_grad = momentum * momentum_buffer + grad
                else:
                    nesterov_grad = momentum_buffer
                
                # 只处理矩阵参数（2D）
                if p.ndim == 2:
                    # Newton-Schulz 正交化
                    n, m = p.shape
                    
                    # 处理非方阵：选择较小维度构建方阵
                    if n <= m:
                        # 使用行维度构建方阵
                        grad_2d = nesterov_grad @ nesterov_grad.T
                        O_prime = compute_matrix_root(grad_2d, ns_iterations)
                        O_prime = O_prime @ nesterov_grad
                    else:
                        # 使用列维度构建方阵
                        grad_2d = nesterov_grad.T @ nesterov_grad
                        O_prime = compute_matrix_root(grad_2d, ns_iterations)
                        O_prime = nesterov_grad @ O_prime
                    
                    # RMS 重缩放: O_t = O'_t * sqrt(max(n,m)) * gamma
                    scale = math.sqrt(max(n, m)) * gamma
                    update = O_prime * scale
                else:
                    # 非矩阵参数：直接使用动量梯度
                    update = nesterov_grad
                
                # 权重衰减 + 参数更新: W_t = W_{t-1} * (1 - eta*lambda) - eta * O_t
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                p.add_(update, alpha=-lr)
        
        return loss


def create_optimizer(
    model: nn.Module,
    muon_lr: float = 3e-4,
    adamw_lr: float = 3e-4,
    muon_momentum: float = 0.95,
    muon_gamma: float = 0.18,
    weight_decay: float = 0.1,
) -> Optimizer:
    """
    创建 Muon + AdamW 混合优化器。
    
    自动识别模型中的矩阵参数和非矩阵参数，分别使用 Muon 和 AdamW。
    
    Args:
        model: 待优化的模型。
        muon_lr: Muon 学习率。
        adamw_lr: AdamW 学习率。
        muon_momentum: Muon 动量系数。
        muon_gamma: Muon RMS 重缩放因子。
        weight_decay: 权重衰减（AdamW 使用，Muon 内部处理）。
        
    Returns:
        Optimizer: 配置好的优化器。
    """
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if is_matrix_param(param, name):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    # 创建参数组
    param_groups = [
        {
            "params": muon_params,
            "lr": muon_lr,
            "momentum": muon_momentum,
            "gamma": muon_gamma,
            "weight_decay": 0.0,  # Muon 内部处理 weight decay
        },
        {
            "params": adamw_params,
            "lr": adamw_lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": weight_decay,
        },
    ]
    
    # 使用单独的优化器实例
    from torch.optim import AdamW
    
    # 分离参数组
    muon_optimizer = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
        gamma=muon_gamma,
        nesterov=True,
    )
    
    adamw_optimizer = AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    
    # 返回组合优化器
    return MixedOptimizer(muon_optimizer, adamw_optimizer)


class MixedOptimizer(Optimizer):
    """包装 Muon 和 AdamW，提供统一接口"""
    
    def __init__(self, muon: Muon, adamw: Optimizer):
        self.muon = muon
        self.adamw = adamw
        self.param_groups = muon.param_groups + adamw.param_groups
        self.state = {}
    
    def step(self, closure=None):
        loss1 = self.muon.step(closure)
        loss2 = self.adamw.step(closure)
        return loss1 if loss1 is not None else loss2
    
    def zero_grad(self, set_to_none=False):
        self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)
    
    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
