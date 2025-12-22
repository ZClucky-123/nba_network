import pandas as pd
import itertools
from collections import Counter
import os


def generate_nba_network_final(file_path, output_file='nba_active_player_edges.csv'):
    print(f"1. 正在读取文件: {file_path} ...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return

    # --- 数据清洗 ---
    print("2. 正在清洗数据格式...")
    # 统一转为字符串并去除首尾空格（防止 "Name " 和 "Name" 不匹配）
    df['Player'] = df['Player'].astype(str).str.strip()
    df['Tm'] = df['Tm'].astype(str).str.strip()

    # 确保年份是数字，处理可能的格式错误
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)

    # --- 确定现役球员逻辑 ---
    # 逻辑：只要在最近两个赛季（例如2025或2026）出现过，即视为现役
    # 这样可以防止因伤缺席最新赛季的球员被漏掉
    max_year = df['Year'].max()
    target_years = [max_year, max_year - 1]

    print(f"   数据集最新年份: {max_year}")
    print(f"   判定现役标准: 在 {target_years} 任意一年有记录")

    active_df = df[df['Year'].isin(target_years)]
    active_players = set(active_df['Player'].unique())
    print(f"   成功识别出 {len(active_players)} 名现役球员。")

    # --- 过滤无效球队行 ---
    # 去除 'TOT' (Total), '2TM' 等汇总行，只保留具体球队
    # 这样确保球员是在具体的更衣室里与队友产生联系
    invalid_teams = ['TOT', '2TM', '3TM', '4TM', '5TM']
    df_clean = df[~df['Tm'].isin(invalid_teams)].copy()

    # --- 构建网络 ---
    print("3. 正在构建队友关系网络...")
    edge_weights = Counter()

    # 按 (年份, 球队) 分组，确保只有“同年同队”才算队友
    grouped = df_clean.groupby(['Year', 'Tm'])

    for (year, team), group in grouped:
        roster = group['Player'].unique()

        # 筛选：只保留现役球员之间的关系
        # 如果你希望包含退役球员的历史关系，可以去掉这个 if 判断
        current_active_roster = [p for p in roster if p in active_players]
        current_active_roster.sort()  # 排序，确保 (A,B) 和 (B,A) 算作同一条边

        if len(current_active_roster) < 2:
            continue

        # 生成两两组合
        for p1, p2 in itertools.combinations(current_active_roster, 2):
            edge_weights[(p1, p2)] += 1

    # --- 导出结果 ---
    print("4. 正在导出数据...")
    edges_data = []
    for (p1, p2), weight in edge_weights.items():
        edges_data.append({'Source': p1, 'Target': p2, 'Weight': weight})

    edges_df = pd.DataFrame(edges_data)

    if edges_df.empty:
        print("错误：生成的边列表为空！请检查数据源。")
        return

    # 按权重降序排列（默契度高的在前）
    edges_df = edges_df.sort_values(by='Weight', ascending=False)

    edges_df.to_csv(output_file, index=False)
    print("-" * 30)
    print(f"✅ 成功！文件已保存至: {output_file}")
    print(f"   共包含 {len(edges_df)} 条队友关系边。")
    print(f"   前 5 对默契度最高的组合:\n{edges_df.head()}")


if __name__ == "__main__":
    # 请确保这里的 CSV 文件名和你文件夹里的一致
    INPUT_CSV = 'NBA_Rosters_2002_2026.csv'
    generate_nba_network_final(INPUT_CSV)