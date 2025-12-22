import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from scipy.sparse import csr_matrix


# --- 1. 特征提取工具函数 (源自参考代码 A) ---
def extract_features(G, edges):
    """
    为给定的边提取拓扑特征：
    1. Degree Difference (度差)
    2. Clustering Coefficient Difference (聚类系数差)
    3. Common Neighbors (共同邻居数量)
    """
    data = []

    # 预计算聚类系数和度
    clustering = nx.clustering(G)
    degrees = dict(G.degree())

    # 转换为邻接矩阵以快速计算共同邻居
    nodes_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(nodes_list)}
    adj = nx.to_scipy_sparse_array(G, nodelist=nodes_list)
    # A^2 的元素 (i, j) 就是节点 i 和 j 的共同邻居数
    common_neighbors_mat = adj @ adj

    for u, v in edges:
        if u not in G or v not in G:
            continue

        u_idx, v_idx = node_map[u], node_map[v]

        # 特征 1: 度差异
        deg_diff = abs(degrees[u] - degrees[v])

        # 特征 2: 聚类系数差异
        clus_diff = abs(clustering[u] - clustering[v])

        # 特征 3: 共同邻居 (从矩阵中获取)
        cn = common_neighbors_mat[u_idx, v_idx]

        data.append([deg_diff, clus_diff, cn])

    return np.array(data)


# --- 2. 机器学习优化权重 (源自参考代码 A/B 的逻辑) ---
def refine_graph_with_ml(G, initial_communities):
    print("   [ML] 正在提取拓扑特征并训练模型...")

    # 1. 生成标签 (Label Generation)
    # 如果两个节点在初始社区中属于同一组，Label=1，否则 Label=0
    node_to_comm = {}
    for idx, comm in enumerate(initial_communities):
        for node in comm:
            node_to_comm[node] = idx

    # 2. 准备训练数据
    # 我们使用图现有的边作为样本
    edges = list(G.edges())
    X = extract_features(G, edges)

    # y 是 "伪标签" (Pseudo-labels)，基于初始划分
    y = []
    for u, v in edges:
        comm_u = node_to_comm.get(u, -1)
        comm_v = node_to_comm.get(v, -2)
        y.append(1 if comm_u == comm_v else 0)

    # 3. 训练 XGBoost 模型
    # 这里我们简化流程，不做复杂的 GridSearchCV，直接用强分类器
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    clf.fit(X, y)

    accuracy = clf.score(X, y)
    print(f"   [ML] 模型自监督学习准确率: {accuracy:.2f}")

    # 4. 重新赋权 (Reweighting)
    # 预测两条边属于同一社区的概率 (Probability)
    probs = clf.predict_proba(X)[:, 1]

    G_refined = G.copy()

    # 结合策略：新权重 = 原始权重 * 机器学习预测概率^2
    # 这样既保留了历史合作时长，又引入了结构合理性判断
    for i, (u, v) in enumerate(edges):
        old_weight = G[u][v].get('Weight_Cubed', 1)  # 获取之前的三次方权重
        prediction_factor = probs[i] * probs[i]

        # 避免权重归零，设置极小值
        if prediction_factor == 0:
            prediction_factor = 1e-6

        new_weight = old_weight * prediction_factor
        G_refined[u][v]['weight'] = new_weight  # 更新用于社区检测的 'weight'

    print("   [ML] 图权重已根据模型预测进行修正。")
    return G_refined


# --- 3. 主程序 (集成你的可视化逻辑) ---
def analyze_nba_communities_ml_enhanced(input_file='nba_active_player_edges.csv'):
    print(f"1. 读取数据: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("错误: 找不到文件。")
        return

    # 应用你原来的三次方策略作为 "Base Feature"
    df['Weight_Cubed'] = df['Weight'] ** 3
    print("   已应用基础权重策略 (Weight^3)。")

    # 构建初始图
    G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr=['Weight', 'Weight_Cubed'])

    # --- 第一阶段：初始社区发现 (粗分) ---
    print("2. 正在进行初始社区划分 (Pre-training Partition)...")
    try:
        # 使用 weight='Weight_Cubed' 进行初步划分
        initial_communities = greedy_modularity_communities(G, weight='Weight_Cubed', resolution=1.2)
    except TypeError:
        initial_communities = greedy_modularity_communities(G, weight='Weight_Cubed')

    print(f"   初始拆解出 {len(initial_communities)} 个社区。")

    # --- 第二阶段：机器学习精修 (ML Refinement) ---

    # 这里我们调用上面定义的函数，用 XGBoost 学习拓扑结构，修正不合理的边
    G_refined = refine_graph_with_ml(G, initial_communities)

    # --- 第三阶段：基于新权重的最终划分 ---
    print("3. 正在进行最终高精度社区划分 (Final Detection)...")
    final_communities = greedy_modularity_communities(G_refined, weight='weight', resolution=1.2)
    print(f"   最终优化为 {len(final_communities)} 个独立社区。")

    # --- 可视化与导出 (保持你的原始风格) ---
    community_data = []
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    plt.figure(figsize=(16, 16))
    # 使用 spring_layout 布局，基于新权重，会让关系紧密的点靠得更近
    pos = nx.spring_layout(G_refined, seed=42, k=0.15, weight='weight')

    print("\n--- 最终社区预览 (ML Optimized) ---")

    for i, community in enumerate(final_communities):
        members = list(community)
        # 找核心 (这里我们还是用原始的三次方权重来决定谁是老大，因为这代表历史地位)
        members_sorted = sorted(members,
                                key=lambda x: G.degree(x, weight='Weight_Cubed'),
                                reverse=True)
        top_5 = members_sorted[:5]

        if i < 15:
            print(f"Group {i + 1} (Size: {len(members)}): {', '.join(top_5)}")

        for player in members_sorted:
            community_data.append({
                'Player': player,
                'Group_ID': i + 1,
                'Role': 'Core' if player in top_5 else 'Member',
                'Color_Index': i
            })

        color = [colors[i % len(colors)]]

        # 绘制核心节点
        core_nodes = [p for p in members if p in top_5]
        if core_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_size=300,
                                   node_color=color, alpha=0.9, label=f"Group {i + 1}")

        # 绘制普通成员节点
        member_nodes = [p for p in members if p not in top_5]
        if member_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=member_nodes, node_size=60,
                                   node_color=color, alpha=0.8)

    # 绘图连线 (过滤弱连接，让图更清晰)
    edges_to_draw = []
    widths = []
    for u, v, d in G.edges(data=True):  # 注意：这里用原图 G 来画线，因为我们想看的是真实的合作关系
        w = d['Weight']
        if w >= 1:
            edges_to_draw.append((u, v))
            widths.append(w * 0.25)

    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, width=widths, alpha=0.2, edge_color='gray')

    plt.title("NBA Active Player Communities (ML-Refined with XGBoost)", fontsize=15)
    plt.axis('off')

    plt.savefig('nba_communities_ml_optimized.png', bbox_inches='tight', dpi=300)
    print("图片已保存: nba_communities_ml_optimized.png")

    df_out = pd.DataFrame(community_data)
    df_out.to_csv('nba_player_communities_ml_optimized.csv', index=False)
    print("名单已保存: nba_player_communities_ml_optimized.csv")


if __name__ == "__main__":
    analyze_nba_communities_ml_enhanced()