# DeepSeek V4 Reproduction

## 项目目录结构

```
deepseek-v4-reproduction/
├── .env                          # 环境变量配置文件
├── .gitignore                    # Git忽略文件配置
├── DeepSeek_V4.pdf               # DeepSeek V4论文PDF
├── README.md                     # 项目说明文档
├── requirements.txt              # Python依赖包列表
├── config/                       # 配置文件目录
├── inference/                    # 推理相关代码
│   ├── generate.py               # 文本生成脚本
│   └── model.py                  # 推理模型实现
├── logs/                         # 日志文件目录
│   └── .gitkeep
├── outputs/                      # 输出文件目录
│   └── .gitkeep
├── plan_iteration/               # 规划迭代文档
│   ├── overall_strategy/         # 整体策略文档
│   │   └── strategy-v1.md        # 策略版本1
│   └── plan01.md                 # 规划
├── pretraining_weights/          # 预训练权重
│   └── stage_1/                  # 第一阶段预训练权重
├── pretrain_dataset/             # 预训练数据集
│   └── .gitkeep
├── source_code/                  # deepseek_v4源码
├── src/                          # 源代码目录
│   ├── data/                     # 数据处理模块
│   ├── models/                   # 模型定义模块
│   │   ├── block.py              # Transformer块实现
│   │   ├── config.py             # 模型配置
│   │   ├── layers.py             # 神经网络层实现
│   │   ├── mhc_attention.py      # MHC注意力机制
│   │   ├── moe.py                # 专家混合(MoE)模块
│   │   ├── rotary_embedding.py   # 旋转位置编码
│   │   └── transformer.py        # Transformer模型主体
│   └── training/                 # 训练相关模块
├── tools/                        # 工具脚本目录
└── utils/                        # 工具函数目录
```

## 项目简介

本项目旨在复现 DeepSeek V4 模型，基于论文实现相关的模型架构和训练流程。

## 模块说明

- **src/models/**: 核心模型实现，包括Transformer架构、MoE模块、MHC注意力机制等
- **inference/**: 模型推理和生成代码
- **plan_iteration/**: 项目规划和策略文档
- **config/**: 配置文件存放目录
- **tools/**: 各种辅助工具脚本
- **utils/**: 通用工具函数
