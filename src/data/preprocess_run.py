from src.data.config import DataConfig
from src.data.pipeline import DataPipeline

def main():
    # 创建配置
    config = DataConfig(
        base_data_dir="datasets/stage_1_datasets",
        chinese_ratio=0.6,
        english_ratio=0.4
    )

    # 运行流水线
    pipeline = DataPipeline(config)

    # 运行单个阶段
    # pipeline.run_stage(0)  # 下载
    # pipeline.run_stage(1) # html清洗，文本规范化
    # pipeline.run_stage(2) # 去重
    # pipeline.run_stage(3) # test / val / train 切分
    pipeline.run_stage(4) # tokenize

    # 运行所有阶段
    # pipeline.run_all(start_stage=0)

if __name__ == "__main__":
    main()