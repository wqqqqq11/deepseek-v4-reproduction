from src.data.config import DataConfig
from src.data.pipeline import DataPipeline

# 创建配置
config = DataConfig(
    base_data_dir="datasets/stage_1_datasets",
    chinese_ratio=0.6,
    english_ratio=0.4
)

# 运行流水线
pipeline = DataPipeline(config)

# 运行单个阶段
pipeline.run_stage(0)  # 下载

# 或运行所有阶段
# pipeline.run_all(start_stage=0)