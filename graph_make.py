import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities


def analyze_nba_communities_split_visualized(input_file='nba_active_player_edges.csv'):
    print(f"1. 读取数据: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("错误: 找不到文件。")
        return

    # --- 核心修改：使用三次方权重 ---
    df['Weight_Cubed'] = df['Weight'] ** 3
    print("   已应用权重立方策略 (Weight^3)，强行拆分弱联系。")

    # 构建图
    G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr=['Weight', 'Weight_Cubed'])

    # 社区发现
    print("2. 正在进行高精度社区划分...")
    try:
        communities = greedy_modularity_communities(G, weight='Weight_Cubed', resolution=1.2)
    except TypeError:
        communities = greedy_modularity_communities(G, weight='Weight_Cubed')

    print(f"   成功拆解出 {len(communities)} 个独立社区。")

    # 准备导出
    community_data = []
    cmap = plt.get_cmap('tab20')  # 颜色板
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(G, seed=42, k=0.12)  # 稍微拉开一点距离

    print("\n--- 新拆分出的社区预览 ---")

    for i, community in enumerate(communities):
        members = list(community)
        # 找核心 (按立方权重排序)
        members_sorted = sorted(members,
                                key=lambda x: G.degree(x, weight='Weight_Cubed'),
                                reverse=True)
        top_5 = members_sorted[:5]  # 前5名为核心

        # 打印验证
        if i < 15:
            print(f"Group {i + 1} (Size: {len(members)}): {', '.join(top_5)}")

        # 记录数据
        for player in members_sorted:
            community_data.append({
                'Player': player,
                'Group_ID': i + 1,
                'Role': 'Core' if player in top_5 else 'Member',
                'Color_Index': i
            })

        # --- 绘图节点 (修改部分：核心节点变大) ---
        color = [colors[i % len(colors)]]

        # 1. 绘制核心节点 (大尺寸)
        core_nodes = [p for p in members if p in top_5]
        if core_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_size=300,
                                   node_color=color, alpha=0.9, label=f"Group {i + 1}")

        # 2. 绘制普通成员节点 (小尺寸)
        member_nodes = [p for p in members if p not in top_5]
        if member_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=member_nodes, node_size=60,
                                   node_color=color, alpha=0.8)

    # 绘图连线
    edges = []
    widths = []
    for u, v, d in G.edges(data=True):
        w = d['Weight']
        if w >= 1:  # 过滤掉 1 年的偶然队友
            edges.append((u, v))
            widths.append(w * 0.25)

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, alpha=0.2, edge_color='gray')

    plt.title("NBA Active Player Communities (Split: Celtics & Grizzlies Separated)", fontsize=15)
    plt.axis('off')

    plt.savefig('nba_communities_amplified_3.png', bbox_inches='tight', dpi=300)
    print("图片已保存: nba_communities_amplified_2.png")

    df_out = pd.DataFrame(community_data)
    df_out.to_csv('nba_player_communities_amplified_2.csv', index=False)
    print("名单已保存: nba_player_communities_amplified_2.csv")


if __name__ == "__main__":
    analyze_nba_communities_split_visualized()