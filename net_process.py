import pandas as pd
import itertools
from collections import Counter


def generate_teammate_network(file_path, output_file='nba_active_player_edges.csv'):
    print(f"正在读取文件: {file_path} ...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return

    # 1. 确定谁是“现役球员”
    # 逻辑：只要在数据集的“最新年份”出现过的球员，都算现役
    # 注意：这里自动获取数据集中最大的年份
    max_year = df['Year'].max()
    print(f"数据集最新赛季年份: {max_year}")

    active_players = set(df[df['Year'] == max_year]['Player'].unique())
    print(f"识别出 {len(active_players)} 名现役球员 (在 {max_year} 赛季有记录)")

    # 2. 数据预处理
    # 排除无效的球队标签 (如 TOT=Total, 2TM=Two Teams等)
    invalid_teams = ['TOT', '2TM', '3TM', '4TM']
    # 仅保留有效球队的记录
    df_clean = df[~df['Tm'].isin(invalid_teams)]

    # 3. 构建边和权重
    print("正在构建队友关系网络...")
    edge_weights = Counter()

    # 按年份和球队分组，处理每一支具体的球队名单
    # 这样可以确保只有同年同队的才算队友
    grouped = df_clean.groupby(['Year', 'Tm'])

    for (year, team), group in grouped:
        # 获取该队该年的所有球员名单
        roster = group['Player'].unique()

        # 筛选：只保留我们在第1步定义的“现役球员”
        # 如果你希望包含退役球员，可以把这就行注释掉
        current_active_roster = [p for p in roster if p in active_players]

        # 排序：确保 (PlayerA, PlayerB) 和 (PlayerB, PlayerA) 被视为同一条边
        current_active_roster.sort()

        # 两两组合 (Combinations)
        # 如果一个队里有 [A, B, C]，会生成 (A,B), (A,C), (B,C)
        for p1, p2 in itertools.combinations(current_active_roster, 2):
            edge_weights[(p1, p2)] += 1

    # 4. 转换为 DataFrame 并保存
    print("正在导出数据...")
    edges_data = []
    for (p1, p2), weight in edge_weights.items():
        edges_data.append({'Source': p1, 'Target': p2, 'Weight': weight})

    edges_df = pd.DataFrame(edges_data)

    # 按权重降序排列（默契度最高的排前面）
    edges_df = edges_df.sort_values(by='Weight', ascending=False)

    edges_df.to_csv(output_file, index=False)
    print("-" * 30)
    print(f"成功！已生成边列表文件: {output_file}")
    print(f"共包含 {len(edges_df)} 条关系边。")
    print("\n前 5 对默契度最高的现役组合:")
    print(edges_df.head())


if __name__ == "__main__":
    # --- 配置区域 ---
    # 修改这里为你实际的 csv 文件名
    INPUT_CSV = 'NBA_Rosters_2002_2026.csv'

    generate_teammate_network(INPUT_CSV)