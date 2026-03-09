import os
import argparse
import numpy as np
from scipy import sparse

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def analyze_graph_for_gat(data_dir):
    qs_path = os.path.join(data_dir, 'qs_table.npz')
    if not os.path.exists(qs_path):
        print(f"Error: {qs_path} 不存在。请检查数据目录。")
        return

    # 拉取稀疏矩阵
    qs_table = sparse.load_npz(qs_path)
    # 假设 qs_table 的 shape 是 (num_q, num_s)
    num_q, num_s = qs_table.shape
    num_edges = qs_table.nnz

    print_separator("1. 基础图结构统计")
    print(f"题目数量 (Num Questions): {num_q}")
    print(f"知识点数量 (Num Skills):   {num_s}")
    print(f"边总数 (Num Edges Q-S):    {num_edges}")
    
    density = num_edges / (num_q * num_s) if num_q * num_s > 0 else 0
    print(f"二分图整体密度:           {density:.6%}")

    # ========================================================
    # 题目的度数分布 (Q -> S)
    # ========================================================
    print_separator("2. 题目维度的度数分布 (Question -> Skill)")
    # 每道题关联几个知识点
    q_degrees = np.array(qs_table.sum(axis=1)).flatten()
    
    # 过滤掉孤立点 (Degree = 0)
    valid_q_degrees = q_degrees[q_degrees > 0]
    num_valid_q = len(valid_q_degrees)
    
    print(f"平均每道题关联知识点数: {valid_q_degrees.mean():.4f}")
    print(f"最大知识点关联数:       {valid_q_degrees.max()}")
    
    q_deg_1_count = np.sum(valid_q_degrees == 1)
    q_deg_1_ratio = q_deg_1_count / num_valid_q if num_valid_q > 0 else 0
    print(f"⚠️ [关键指标] 仅关联 1 个知识点的题目比例: {q_deg_1_ratio:.2%} ({q_deg_1_count}/{num_valid_q})")
    
    # ========================================================
    # 知识点的度数分布 (S -> Q)
    # ========================================================
    print_separator("3. 知识点维度的度数分布 (Skill -> Question)")
    # 每个知识点包含多少道题
    s_degrees = np.array(qs_table.sum(axis=0)).flatten()
    
    # 过滤掉孤立点
    valid_s_degrees = s_degrees[s_degrees > 0]
    num_valid_s = len(valid_s_degrees)
    
    if num_valid_s > 0:
        print(f"平均每个知识点包含题目数: {valid_s_degrees.mean():.2f}")
        print(f"中位数 (Median):          {np.median(valid_s_degrees)}")
        print(f"最大题目包含数:           {valid_s_degrees.max()}")
        
        # 对于GAT来说，度数小（连线少）是没有意义的
        s_deg_low_count = np.sum(valid_s_degrees <= 3)
        s_deg_low_ratio = s_deg_low_count / num_valid_s
        print(f"⚠️ [关键指标] 包含题目数 ≤ 3 的知识点比例: {s_deg_low_ratio:.2%} ({s_deg_low_count}/{num_valid_s})")
        
        # 高度数节点，Attention能发挥作用的区别度高
        s_deg_high_count = np.sum(valid_s_degrees >= 20)
        s_deg_high_ratio = s_deg_high_count / num_valid_s
        print(f"🌟 [GAT潜力区] 包含题目数 ≥ 20 的知识点比例: {s_deg_high_ratio:.2%} ({s_deg_high_count}/{num_valid_s})")

    # ========================================================
    # 诊断与结论
    # ========================================================
    print_separator("4. GAT vs GCN 结构诊断报告")
    
    degeneration_score = 0
    reasons = []

    # 诊断 1: GAT 的数学必然退化 (Softmax 1 = 1.0)
    if q_deg_1_ratio > 0.9:
        degeneration_score += 50
        reasons.append(f"【数学性退化】高达 {q_deg_1_ratio:.1%} 的题目只映射了1个知识点！在计算注意力时，Softmax(1个元素) 永远等于 1.0。这意味着在这部分图传播上，GAT计算了一堆复杂的权重矩阵，最终只能得出与 GCN（或均值聚合）完全相同的固定权重1.0，白白浪费了计算量和参数。")
    elif q_deg_1_ratio > 0.5:
        degeneration_score += 30
        reasons.append(f"【边际效益低】超过一半 ({q_deg_1_ratio:.1%}) 的题目只映射1个知识点。在这些节点上，GAT等价于GCN。")

    # 诊断 2: GAT 的过拟合风险 (样本量极少网络)
    if num_valid_s > 0 and s_deg_low_ratio > 0.3:
        degeneration_score += 30
        reasons.append(f"【严重过拟合风险】{s_deg_low_ratio:.1%} 的知识点只包含不到3道题。在极少邻居（如2个）进行 Attention 时，模型极易退化成只盯着（权重大于0.99）偶然表现好的那道题，而抛弃了原本起正则化作用的均值结构。")

    # 诊断 3: GAT 的用武之地 (高自由度聚类)
    has_gat_potential = False
    if num_valid_s > 0 and s_deg_high_ratio > 0.5:
        has_gat_potential = True
        reasons.append(f"【极高GAT潜力】有 {s_deg_high_ratio:.1%} 的知识点包含超过20道题目。对包含数百个题目的泛化大知识点，GAT 的表现往往远超 GCN，它能聪明地挑出哪些题目最能代表该知识点，哪些可能只是噪音。")

    # 综合评估
    print("\n[最终结论判定]：")
    if degeneration_score >= 50 and not has_gat_potential:
        print("❌ 极不推荐 GAT。表现几乎不可能超过 GCN (甚至更差且更慢)。")
    elif degeneration_score >= 30 and has_gat_potential:
        print("⚠️ 混合表现。总分可能没有明显提升，但 GAT 可能帮助那部分密集的节点（知识点）取得了更好特征。")
    elif degeneration_score < 30 and has_gat_potential:
        print("✅ 强烈推荐 GAT！因为度数丰富，GAT有充分施展注意力的空间。")
    else:
        print("🤷‍♂️ 处于中间地带，表现与 GCN 应该相近 (微弱提升或微弱下降)。")

    print("\n[具体原因]：")
    for i, r in enumerate(reasons, 1):
        print(f"  {i}. {r}")
        
    print("\\n💡 [论文写作建议]: 无论结果如何，都可以把这份指标写进论文（比如图稀疏度 sparsity，平均度数等指标），用来论证模型设计/消融实验合理性：\\n   '由于目标数据集中的Q-S图呈现高度单射特性（XX%题目仅映射1个知识点），注意力机制易产生不可避免的数学退化...'")

# python scripts/analyze_graph_for_gat.py --data_dir back/kt/data/ednet_kt1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"H:/er_gikt/back/kt/data/ednet_kt1", help="Path to processed data directory")
    args = parser.parse_args()
    
    analyze_graph_for_gat(args.data_dir)