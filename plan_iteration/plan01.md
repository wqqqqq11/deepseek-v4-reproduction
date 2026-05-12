# 阶段一
## 数据集预处理
1. 下载 Fineweb 中文数据、RedPajama 英文数据

2. 原始字段抽取
   只取 text 字段

3. 基础清洗
   去 HTML
   去 URL
   去广告
   去异常符号
   去空文本
   去过短文本
   去超长异常文本

4. 质量过滤
   中文数据：中文字符比例过低的丢弃
   英文数据：英文字符比例过低的丢弃
   标点/数字/乱码比例异常的丢弃

5. 去重
   先精确去重
   有精力再做近似去重

6. 文档级划分
   train / val / test

7. tokenize
   每篇文档 encode 成 token ids
   每篇文档末尾加 eos_token_id

8. 拼接
   train 文档拼成 train_token_stream
   val 文档拼成 val_token_stream
   test 文档拼成 test_token_stream

9. 切块
   每 4097 个 token 切一块
   x = 前 4096
   y = 后 4096

10. 训练
   model(x)
   causal LM loss
