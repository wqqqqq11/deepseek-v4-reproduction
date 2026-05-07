# transformer blocke 解析
1、MoE 详解
1.1 总述
    每一个transformer_block都有一个MoE，1 个 Shared Expert + 384 个 Routed Expert（N_s = 1, N_r = 384）
1.2 作用
1.3 组件
定义：全员必走的通用处理 独立于路由机制，所有 token 都必须经过的 “基础专家”，提供全局通用特征提取。

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