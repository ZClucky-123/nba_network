import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

# 引入高级机器学习库 (参考代码 2)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from scipy.sparse import csr_matrix


# --- 1. 特征提取工具函数 (保持不变，效果很好) ---
def extract_features(G, edges):
    """
    提取拓扑特征：度差、聚类系数差、共同邻居
    """
    data = []
    clustering = nx.clustering(G)
    degrees = dict(G.degree())

    nodes_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(nodes_list)}
    adj = nx.to_scipy_sparse_array(G, nodelist=nodes_list)
    common_neighbors_mat = adj @ adj

    for u, v in edges:
        if u not in G or v not in G:
            continue
        u_idx, v_idx = node_map[u], node_map[v]
        deg_diff = abs(degrees[u] - degrees[v])
        clus_diff = abs(clustering[u] - clustering[v])
        cn = common_neighbors_mat[u_idx, v_idx]
        data.append([deg_diff, clus_diff, cn])

    return np.array(data)


# --- 2. 高级机器学习优化权重 (核心升级部分) ---
def refine_graph_with_advanced_ml(G, initial_communities):
    print("   [ML-Advanced] 正在提取特征并使用集成模型 (Ensemble) 进行训练...")

    # 1. 准备数据
    node_to_comm = {}
    for idx, comm in enumerate(initial_communities):
        for node in comm:
            node_to_comm[node] = idx

    edges = list(G.edges())
    X = extract_features(G, edges)

    # 生成伪标签
    y = []
    for u, v in edges:
        comm_u = node_to_comm.get(u, -1)
        comm_v = node_to_comm.get(v, -2)
        y.append(1 if comm_u == comm_v else 0)
    y = np.array(y)

    # 2. 定义 K-Fold 交叉验证 (参考代码 2 的逻辑)
    # 我们使用 5 折交叉验证来生成对每条边的预测概率
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 用于存储所有样本的预测概率 (初始化为 0)
    final_probs = np.zeros(len(y))

    print(f"   [ML-Advanced] 开始 5-Fold 交叉验证与网格搜索 (这可能需要一点时间)...")

    fold_idx = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # --- 模型 A: XGBoost (带网格搜索) ---
        xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_grid = GridSearchCV(
            estimator=xgb_clf,
            param_grid={'max_depth': [3, 5], 'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
            cv=3, scoring='accuracy', n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_

        # --- 模型 B: 随机森林 (带网格搜索) ---
        rf_clf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(
            estimator=rf_clf,
            param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]},
            cv=3, scoring='accuracy', n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_

        # --- 模型 C: 决策树 (带网格搜索) ---
        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_grid = GridSearchCV(
            estimator=dt_clf,
            param_grid={'max_depth': [3, 5], 'min_samples_split': [2, 10]},
            cv=3, scoring='accuracy', n_jobs=-1
        )
        dt_grid.fit(X_train, y_train)
        best_dt = dt_grid.best_estimator_

        # --- 集成: 软投票 (Soft Voting) ---
        voting_clf = VotingClassifier(
            estimators=[('xgb', best_xgb), ('rf', best_rf), ('dt', best_dt)],
            voting='soft'
        )
        voting_clf.fit(X_train, y_train)

        # 对当前 Fold 的测试集进行预测
        probs_fold = voting_clf.predict_proba(X_test)[:, 1]

        # 将预测结果填入对应的索引位置
        final_probs[test_index] = probs_fold

        print(f"     > Fold {fold_idx}/5 完成. 当前批次准确率: {voting_clf.score(X_test, y_test):.2f}")
        fold_idx += 1

    # 3. 重新赋权 (Reweighting)
    G_refined = G.copy()

    # 记录权重的变化情况用于调试
    weight_changes = []

    for i, (u, v) in enumerate(edges):
        old_weight = G[u][v].get('Weight_Cubed', 1)

        # 核心逻辑：新权重 = 旧权重 * (集成模型预测概率 ^ 2)
        prob = final_probs[i]
        prediction_factor = prob * prob

        if prediction_factor < 1e-6:
            prediction_factor = 1e-6

        new_weight = old_weight * prediction_factor
        G_refined[u][v]['weight'] = new_weight

        if i < 5:  # 打印前5个看看
            weight_changes.append(f"Edge ({u}-{v}): {old_weight:.2f} -> {new_weight:.2f} (Prob: {prob:.2f})")

    print("   [ML-Advanced] 权重修正示例:\n     " + "\n     ".join(weight_changes))
    return G_refined


# --- 3. 主程序 (保持原有处理流和可视化) ---
def analyze_nba_communities_ml_enhanced(input_file='nba_active_player_edges.csv'):
    print(f"1. 读取数据: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("错误: 找不到文件。")
        return

    # 保留原有的三次方策略
    df['Weight_Cubed'] = df['Weight'] ** 3
    print("   已应用基础权重策略 (Weight^3)。")

    # 构建初始图
    G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr=['Weight', 'Weight_Cubed'])

    # --- 第一阶段：初始社区发现 ---
    print("2. 正在进行初始社区划分 (Initial Partition)...")
    try:
        initial_communities = greedy_modularity_communities(G, weight='Weight_Cubed', resolution=1.2)
    except TypeError:
        initial_communities = greedy_modularity_communities(G, weight='Weight_Cubed')
    print(f"   初始拆解出 {len(initial_communities)} 个社区。")

    # --- 第二阶段：机器学习精修 (使用新的高级函数) ---
    G_refined = refine_graph_with_advanced_ml(G, initial_communities)

    # --- 第三阶段：最终划分 ---
    print("3. 正在进行最终社区划分 (Final Detection)...")
    final_communities = greedy_modularity_communities(G_refined, weight='weight', resolution=1.2)
    print(f"   最终优化为 {len(final_communities)} 个独立社区。")

    # --- 可视化与导出 (完全保留你的原始逻辑) ---
    community_data = []
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(G_refined, seed=42, k=0.15, weight='weight')

    print("\n--- 最终社区预览 (Ensemble ML Optimized) ---")

    for i, community in enumerate(final_communities):
        members = list(community)
        # 依然使用 Weight_Cubed 来确定核心球员（历史地位）
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

        # 绘点
        core_nodes = [p for p in members if p in top_5]
        if core_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_size=300,
                                   node_color=color, alpha=0.9, label=f"Group {i + 1}")
        member_nodes = [p for p in members if p not in top_5]
        if member_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=member_nodes, node_size=60,
                                   node_color=color, alpha=0.8)

    # 绘线
    edges_to_draw = []
    widths = []
    for u, v, d in G.edges(data=True):
        w = d['Weight']
        if w >= 1:
            edges_to_draw.append((u, v))
            widths.append(w * 0.25)

    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, width=widths, alpha=0.2, edge_color='gray')

    plt.title("NBA Active Player Communities (Optimized with Voting Ensemble & K-Fold)", fontsize=15)
    plt.axis('off')

    plt.savefig('nba_communities_ensemble_optimized.png', bbox_inches='tight', dpi=300)
    print("图片已保存: nba_communities_ensemble_optimized.png")

    df_out = pd.DataFrame(community_data)
    df_out.to_csv('nba_player_communities_ensemble_optimized.csv', index=False)
    print("名单已保存: nba_player_communities_ensemble_optimized.csv")


if __name__ == "__main__":
    analyze_nba_communities_ml_enhanced()