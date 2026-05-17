"""
训练入口

用法:
    python mian.py --stage 1 --config configs/stage1_pretrain.yaml
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.config import ModelArgs
from src.models.transformer_stage1 import TransformerStage1
from src.training.dataset import TokenDataset
from src.training.muon import create_optimizer
from src.training.lr_scheduler import create_scheduler
from src.training.logger import TrainingLogger, get_gpu_memory
from src.training.utils.util import set_seed, count_parameters, clip_gradients


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_stage1(config):
    cfg = config
    
    # 设置种子
    set_seed(cfg["training"]["seed"])
    
    # 模型配置
    model_args = ModelArgs.stage1()
    model_args.vocab_size = cfg["model"]["vocab_size"]
    model_args.dim = cfg["model"]["dim"]
    model_args.n_layers = cfg["model"]["n_layers"]
    model_args.n_heads = cfg["model"]["n_heads"]
    model_args.max_seq_len = cfg["model"]["max_seq_len"]
    model_args.q_lora_rank = cfg["model"]["q_lora_rank"]
    model_args.kv_lora_rank = cfg["model"]["kv_lora_rank"]
    model_args.qk_nope_head_dim = cfg["model"]["qk_nope_head_dim"]
    model_args.v_head_dim = cfg["model"]["v_head_dim"]
    model_args.head_dim = cfg["model"]["head_dim"]
    
    # 创建模型
    device = torch.device(cfg["hardware"]["device"])
    model = TransformerStage1(model_args).to(device)
    
    total_params, trainable = count_parameters(model)
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 优化器
    opt_cfg = cfg["optimizer"]
    optimizer = create_optimizer(
        model,
        muon_lr=opt_cfg["muon_lr"],
        adamw_lr=opt_cfg["adamw_lr"],
        muon_momentum=opt_cfg["muon_momentum"],
        muon_gamma=opt_cfg["muon_gamma"],
        weight_decay=opt_cfg["weight_decay"],
    )
    
    # 学习率调度
    sched_cfg = cfg["training"]
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=sched_cfg["warmup_steps"],
        peak_steps=sched_cfg["peak_steps"],
        cosine_steps=sched_cfg["cosine_steps"],
        max_lr=sched_cfg["max_lr"],
        min_lr=sched_cfg["min_lr"],
        mtp_initial_weight=cfg["mtp"]["loss_weight"],
        mtp_final_weight=cfg["mtp"]["loss_weight_decay_end"],
    )
    
    # 数据
    data_cfg = cfg["data"]
    train_dataset = TokenDataset(
        data_cfg["train_bin"],
        context_size=data_cfg["context_size"],
        shuffle_seed=data_cfg["shuffle_seed"],
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    
    # 日志
    log_cfg = cfg["logging"]
    logger = TrainingLogger(
        log_cfg["log_dir"],
        log_every_steps=log_cfg["log_every_steps"],
    )
    
    # 训练参数
    num_epochs = cfg["training"]["num_epochs"]
    grad_clip = cfg["training"]["grad_clip"]
    log_every = log_cfg["log_every_steps"]
    
    # 训练循环
    global_step = 0
    
    for epoch in range(num_epochs):
        train_dataset.set_epoch(epoch)
        logger.start_epoch(epoch + 1, len(train_loader))
        
        model.train()
        epoch_loss = 0.0
        
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # 前向
            logits = model(x)
            
            # 计算损失 (next token prediction)
            loss = F.cross_entropy(
                logits.view(-1, model_args.vocab_size),
                y.view(-1),
                ignore_index=-1,
            )
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_norm = clip_gradients(model, grad_clip)
            
            # 更新
            optimizer.step()
            sched_info = scheduler.step()
            
            # 记录
            global_step += 1
            epoch_loss += loss.item()
            
            if step % log_every == 0:
                tokens_per_sec = (
                    cfg["training"]["batch_size"] * data_cfg["context_size"] / 
                    (logger.step_time if hasattr(logger, "step_time") else 1.0)
                )
                
                metrics = {
                    "loss/total": loss.item(),
                    "loss/lm": loss.item(),
                    "lr": sched_info["lr"],
                    "tokens_per_sec": tokens_per_sec,
                    "grad_norm": grad_norm,
                    "gpu_memory": get_gpu_memory(),
                }
                
                logger.log_step(step, metrics, model)
            
            # 打印进度
            if step % (log_every * 10) == 0:
                print(f"  step {step}/{len(train_loader)} | loss: {loss.item():.4f} | lr: {sched_info['lr']:.2e}")
        
        # 计算 epoch 平均损失
        avg_loss = epoch_loss / len(train_loader)
        
        # 验证
        print(f"\n验证 epoch {epoch + 1}...")
        val_loss = evaluate(model, cfg, device, model_args.vocab_size)
        
        logger.log_eval(step, {
            "val/loss": val_loss,
            "val/ppl": torch.exp(torch.tensor(val_loss)).item(),
        })
        
        print(f"Epoch {epoch + 1} 完成 | train_loss: {avg_loss:.4f} | val_loss: {val_loss:.4f}")
        
        # 保存检查点
        if log_cfg["save_best_only"]:
            ckpt_path = Path(log_cfg["checkpoint_dir"]) / "best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": val_loss,
            }, ckpt_path)
        
        logger.end_epoch()
    
    logger.close()
    print("\n训练完成！")


def evaluate(model, cfg, device, vocab_size):
    """验证"""
    data_cfg = cfg["data"]
    
    try:
        val_dataset = TokenDataset(
            data_cfg["val_bin"],
            context_size=data_cfg["context_size"],
            shuffle_seed=data_cfg["shuffle_seed"],
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
    except FileNotFoundError:
        print("  验证数据不存在，跳过")
        return 0.0
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=-1,
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 100:  # 最多验证100个batch
                break
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, help="训练阶段")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    if args.stage != 1:
        print(f"阶段 {args.stage} 暂未实现")
        return
    
    config = load_yaml(args.config)
    train_stage1(config)


if __name__ == "__main__":
    main()
