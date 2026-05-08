# transformer block 解析
deepseek-v4-pro 的每个transformer block都由RMS Norm-->
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
    DeepSeek V4 在每个 Transformer Block 中，对 Attention 子层和 MoE 子层分别使用一次 mHC 残差结构；模型隐藏状态 x_l 并非单一 residual stream，而是在内部新增一个 HC 维度，维护 hc_mult=4 条并行残差流。它们同源于同一个输入 token embedding，不是 4 份不同外部输入，而是形成形如 [B, S, 4, 7168] 的内部多流状态。
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