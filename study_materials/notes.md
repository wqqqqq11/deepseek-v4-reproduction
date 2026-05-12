# transformer block 解析
1 MoE（Mixture of Experts 混合专家） 详解
1.1 总述
    每一个transformer_block都有一个MoE，1 个 Shared Expert + 384 个 Routed Expert（N_s = 1, N_r = 384）
1.2 作用
1.3 组件
定义：全员必走的通用处理 独立于路由机制，所有 token 都必须经过的 “基础专家”，提供全局通用特征提取。
每一个 token 单独自己选 6 个routed专家

1.4 流程
1.4.1 Routed Expert 主线
（1）输入 u_t （RMSNorm 输出）传给 Router
（2）Router 给每个 Routed Expert 打一个分数
（3）这些分数经过 softmax 后，变成权重（标量）（黄色线的来源）
（4）每个被选中的 Routed Expert，输出向量会乘以自己的权重，再和Shared Expert的结果相加得到结果h'_t
1.4.2 Shared Expert 支线
（1）输入 ut​（RMSNorm 输出）直接传入 Shared Expert
（2）每个 Shared Expert 独立计算输出并且相加
（3）Shared 支路的最终结果，和 Routed Expert支路的加权结果相加，得到 MoE 层的总输出h'_t​
1.4.2 Expert 的组成

每个Expert的结构都完全一样，称之为SwiGLU（SwiSH Gated Linear Unit，门控激活函数）。SwiGLU是将老式GLU中的激活函数从Sigmoid换为SiLU（Swish，即图中的Activation组件）演化而来。

2. mHC（Manifold-Constrained Hyper-Connections 流形约束超残差）详解
DeepSeek V4 在每个 Transformer Block 中，对 Attention 子层和 MoE 子层分别使用一次 mHC 残差结构；模型隐藏状态 x_l 并非单一 residual stream，而是在内部新增一个 HC 维度，维护 hc_mult=4 条（以图中4为例）并行残差流。它们同源于同一个输入 token embedding，不是 4 份不同外部输入，而是形成形如 [B, S, 4, 7168] 的内部多流状态。

2.1 第一个 mHC（绑定 Attention 子层）

    主路径：x_l​（4 条并行残差流）→ RMSNorm → Pre Mapping（多流流形前置对齐）→  Attention（Layer F）→ Post Mapping（多流后置流形对齐）→ h_l^post​

    残差路径：x_l​（同源 4 条并行流）→ Res Mapping（残差流专属流形变换）→ h_l^res​

    融合：4 条残差流做流形约束加权融合，输出作为下一段输入 x_(l+1​)


2.2 第二个 mHC（绑定 MoE 子层）

    主路径：x_(l+1)​（4 条并行残差流）→ RMSNorm → Pre Mapping（多流前置对齐）→  MoE Block（Layer F）→ Post Mapping（多流后置流形对齐）→h_l^post​

    残差路径：x_(l+1)​（同源 4 条并行流）→ Res Mapping（残差流专属流形变换）→h_l^res​

    融合：同样对 4 条并行残差流做流形约束加权融合，输出本 Block 最终结果 x_(l+2​)

2.3 文字叙述：
    支路：x_t 维护 4 条 streams，但它们不是完全独立做 Res Mapping，而是通过 comb 矩阵做受约束的加权混合，得到仍然是 4 条 streams 的 h_t^res。
    主路：同样的 x_t 经过 Pre Mapping 被压成一条 stream，即 h_t^in；再经过 RMSNorm + Layer F 得到一条 h_t^out；然后通过 Post Mapping 写回 4 条 streams。
    最后：h_t^res + h_t^post 融合，输出仍然是 4 条 streams。

3. CSA（Compress Sparse Attention）
3.1 CSA 核心定位
CSA 是 DeepSeek-V4混合注意力（CSA+HCA）的核心模块，专为百万 token 长上下文设计：先压缩 KV 缓存→再做稀疏注意力选择，同时保留滑动窗口捕捉局部依赖，最终把长序列的计算量、KV 缓存量压到极低。

3.2 CSA 数据流
一、整体数据流的起点：两个核心输入
整个 CSA 模块的数据流，是从两组输入开始分路处理，最后再汇合计算注意力：

左侧输入：Hidden States of KV Tokens（历史 KV Token 的隐状态）这是序列中所有之前 Token 的隐状态，会分成三条路径处理：

路径 A：直接作为Sliding Window KV Entries（滑动窗口 KV 条目），保留最近 N 个 Token 的原始、未压缩 KV。
路径 B：进入左侧的Token-Level Compressor（Token 级压缩器），生成压缩后的 KV 条目。
路径 C：进入Lightning Indexer内部的Token-Level Compressor，生成索引用的压缩键。


右侧输入：Hidden State of Query Token（当前 Query Token 的隐状态）这是当前要计算注意力的 Token 的隐状态，会分成两条路径处理：

路径 1：生成主注意力的Queries，直接送到最终的注意力模块。
路径 2：生成Indexer Queries（索引器查询），送入Lightning Indexer，用于和压缩后的索引键计算相关性得分。


二、KV 侧的压缩与筛选流程（路径 B）
1. 左侧Token-Level Compressor

输入：Hidden States of KV Tokens（原始 KV 隐状态）
处理：按论文中的 Token 级重叠压缩，将连续 m 个 Token 的隐状态压缩为 1 个向量，大幅减少 KV 的序列长度。
输出：Compressed KV Entries（压缩后的 KV 条目），送入Top-k Selector模块等待筛选。
目的：把长序列 KV 的长度直接压缩为原来的 1/m，降低后续计算的规模。


三、Lightning Indexer：压缩 KV 的相关性筛选（虚线框内）
这是 CSA 的核心筛选机制，用来从压缩后的 KV 里挑出和当前 Query 最相关的 Top-k 个条目，避免计算所有压缩 KV 的注意力。
1. 内部Token-Level Compressor

输入：Hidden States of KV Tokens（原始 KV 隐状态）
处理：和左侧的压缩器逻辑一致，对 KV 隐状态进行压缩，生成索引专用的压缩键。
输出：Compressed Indexer Keys（压缩后的索引键），送入内部的Multi-Query Attention模块。

2. 内部Multi-Query Attention

输入：Compressed Indexer Keys（压缩索引键） + Indexer Queries（来自 Query Token 的索引查询）
处理：计算当前 Query 和每个压缩索引键的注意力得分（即相关性得分），用 MQA（多查询注意力）高效完成计算。
输出：Index Scores（索引得分），送入Top-k Selector模块。

3. Top-k Selector

输入：Compressed KV Entries（来自左侧压缩器的所有压缩 KV） + Index Scores（来自 Lightning Indexer 的相关性得分）
处理：根据 Index Scores，从所有 Compressed KV Entries 中选出得分最高的 Top-k 个条目。
输出：Selected Compressed KV Entries（选中的压缩 KV 条目），送入Concatenation模块。
目的：直接把需要计算注意力的压缩 KV 数量从 “全部压缩量” 降到 “k 个”，大幅减少后续注意力计算的成本。


四、局部 KV 与压缩 KV 的拼接
Concatenation模块

输入：
Sliding Window KV Entries（路径 A 的原始局部 KV，未压缩）
Selected Compressed KV Entries（路径 B+Lightning Indexer 筛选后的压缩 KV）


处理：将两部分 KV 条目拼接起来，形成完整的 KV 集合。
输出：拼接后的 KV 集合，送入最终的Shared Key-Value Multi-Query Attention模块。
目的：同时保留最近的局部细粒度 KV和筛选出的关键长距离压缩 KV，既不丢失细节，也覆盖了长距离依赖。


五、最终注意力计算
Shared Key-Value Multi-Query Attention模块

输入：
Queries（来自 Query Token 的主查询）
拼接后的 KV 集合（Sliding Window KV + Selected Compressed KV）


处理：使用共享 KV 的 MQA（所有注意力头共享同一组 KV，只有 Queries 是多头的），计算当前 Query 和拼接后 KV 的注意力。
输出：最终的注意力输出，即 CSA 模块的结果，送入后续的模型层。
目的：用共享 KV 的方式进一步降低内存和计算成本，同时完成核心的注意力计算。

六、数据流的核心逻辑
整个 CSA 的数据流是 **“压缩 + 筛选 + 局部补全”** 的组合拳：

先压缩：把长序列 KV 压缩成短序列，降低整体规模；
再筛选：用 Lightning Indexer 快速挑出和当前 Query 最相关的压缩 KV，避免计算所有压缩 KV；
局部补全：用滑动窗口保留原始局部 KV，弥补压缩带来的局部信息损失；
高效计算：用共享 KV 的 MQA 做最终注意力，进一步降低成本。

CSA 在处理百万级 Token 的长上下文时，既能大幅降低 KV 缓存和计算量，又不会因为压缩而丢失关键信息，完美平衡了效率和效果。