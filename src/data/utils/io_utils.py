"""IO工具函数"""

import json
import logging
import asyncio
import numpy as np

from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional, AsyncIterator

logger = logging.getLogger(__name__)


try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logger.warning("aiofiles not available, using sync fallback")


try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False


def write_jsonl(filepath: str, records: Iterator[Dict]) -> int:
    """将记录写入 jsonl 文件，返回写入数量"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
                if count % 10000 == 0:
                    logger.info(f"已写入 {count} 条记录到 {path}")
    except IOError as e:
        logger.error(f"写入文件失败 {path}: {e}")
        raise

    logger.info(f"写入完成: {path}, 共 {count} 条记录")
    return count


def read_jsonl(filepath: str) -> Iterator[Dict]:
    """从 jsonl 文件读取记录"""
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"文件不存在: {path}")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"解析失败 {path}:{line_num}: {line[:100]}")
    except IOError as e:
        logger.error(f"读取文件失败 {path}: {e}")
        raise


def write_bin(filepath: str,
              xy_pairs: Iterator[Tuple[List[int], List[int]]],
              dtype=np.uint16) -> int:
    """将 (x, y) 对写入二进制文件，返回写入样本数"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_x = []
    temp_y = []
    count = 0

    try:
        for x, y in xy_pairs:
            temp_x.append(x)
            temp_y.append(y)
            count += 1

            if count % 10000 == 0:
                logger.info(f"已缓冲 {count} 条样本")

        if count == 0:
            logger.warning("没有数据可写入")
            return 0

        x_array = np.array(temp_x, dtype=dtype)
        y_array = np.array(temp_y, dtype=dtype)

        header = np.array([count], dtype=np.int64)
        header.tofile(path)
        x_array.tofile(path, sep='')
        y_array.tofile(path, sep='')

    except Exception as e:
        logger.error(f"写入二进制文件失败 {path}: {e}")
        raise

    logger.info(f"写入完成: {path}, 共 {count} 条样本")
    return count


def memory_map_bin(filepath: str) -> Optional[Tuple[np.memmap, np.memmap]]:
    """内存映射二进制文件，返回 (x_mmap, y_mmap)"""
    path = Path(filepath)
    if not path.exists():
        logger.error(f"文件不存在: {path}")
        return None

    try:
        with open(path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int64, count=1)
            n_samples = int(header[0])

        x_size = n_samples * 4096
        offset = 8

        x_mmap = np.memmap(path, dtype=np.uint16, mode='r',
                           offset=offset, shape=(n_samples, 4096))
        y_mmap = np.memmap(path, dtype=np.uint16, mode='r',
                           offset=offset + x_size * 2, shape=(n_samples, 4096))

        return x_mmap, y_mmap
    except Exception as e:
        logger.error(f"内存映射失败 {path}: {e}")
        return None


def load_checkpoint(base_dir: str) -> Optional[Dict]:
    """加载检查点"""
    path = Path(base_dir) / "checkpoint.json"
    if not path.exists():
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"加载检查点失败: {e}")
        return None


def save_checkpoint(base_dir: str, data: Dict) -> None:
    """保存检查点"""
    path = Path(base_dir) / "checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.error(f"保存检查点失败: {e}")
        raise


def list_jsonl_files(directory: str) -> List[Path]:
    """列出目录下所有 jsonl 文件"""
    path = Path(directory)
    if not path.exists():
        return []
    return sorted(path.glob("*.jsonl"))


async def async_read_jsonl_batches(
    filepath: str,
    batch_size: int = 5000
) -> AsyncIterator[List[Dict]]:
    """异步批量读取jsonl文件"""
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"文件不存在: {path}")
        return

    batch = []
    line_num = 0

    try:
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                async for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if ORJSON_AVAILABLE:
                            record = orjson.loads(line)
                        else:
                            record = json.loads(line)
                        batch.append(record)

                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except json.JSONDecodeError:
                        logger.warning(f"解析失败 {path}:{line_num}")
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if ORJSON_AVAILABLE:
                            record = orjson.loads(line)
                        else:
                            record = json.loads(line)
                        batch.append(record)

                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except json.JSONDecodeError:
                        logger.warning(f"解析失败 {path}:{line_num}")

        if batch:
            yield batch

    except IOError as e:
        logger.error(f"读取文件失败 {path}: {e}")
        raise


async def async_write_jsonl_batches(
    filepath: str,
    batches: AsyncIterator[List[Dict]],
    append: bool = False
) -> int:
    """异步批量写入jsonl文件"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    mode = 'a' if append else 'w' 
    try:
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(path, mode, encoding='utf-8') as f:
                async for batch in batches:
                    for record in batch:
                        if ORJSON_AVAILABLE:
                            line = orjson.dumps(
                                record,
                                option=orjson.OPT_APPEND_NEWLINE
                            ).decode('utf-8')
                        else:
                            line = json.dumps(record, ensure_ascii=False) + '\n'
                        await f.write(line)
                        count += 1

                    if count % 10000 == 0:
                        logger.info(f"已写入 {count} 条记录到 {path}")
        else:
            with open(path, mode, encoding='utf-8') as f:
                async for batch in batches:
                    for record in batch:
                        if ORJSON_AVAILABLE:
                            line = orjson.dumps(
                                record,
                                option=orjson.OPT_APPEND_NEWLINE
                            ).decode('utf-8')
                        else:
                            line = json.dumps(record, ensure_ascii=False) + '\n'
                        f.write(line)
                        count += 1

                    if count % 10000 == 0:
                        logger.info(f"已写入 {count} 条记录到 {path}")

        logger.info(f"写入完成: {path}, 共 {count} 条记录")
        return count

    except IOError as e:
        logger.error(f"写入文件失败 {path}: {e}")
        raise
