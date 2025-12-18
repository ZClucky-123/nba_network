import pandas as pd
import os


def process_nba_rosters(file_path):
    print(f"1. 正在读取文件: {file_path}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件。请确认路径是否正确：\n{file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # --- 2. 数据清洗与筛选 ---

    # 根据截图，年份列叫 'season' (整数类型，如 2011)
    # 我们筛选 2002 年及以后的赛季
    print("2. 正在筛选 2002 年以后的数据...")
    df_roster = df[df['season'] >= 2002].copy()

    # 筛选联盟：截图中有 'lg' 列，为了保险起见，只保留 'NBA'
    if 'lg' in df_roster.columns:
        df_roster = df_roster[df_roster['lg'] == 'NBA']

    # 去除交易汇总行：在这个数据集中，球队缩写列叫 'team'
    # 通常 'TOT' 表示 Total (被交易球员的汇总数据)，我们需要具体的球队
    if 'team' in df_roster.columns:
        df_roster = df_roster[df_roster['team'] != 'TOT']

    # --- 3. 整理列名 ---

    # 截图显示的列名 -> 我们想要的标准列名
    # season -> Year
    # team   -> Tm
    # player -> Player
    # pos    -> Pos
    # age    -> Age

    cols_map = {
        'season': 'Year',
        'team': 'Tm',
        'player': 'Player',
        'pos': 'Pos',
        'age': 'Age'
    }

    # 提取需要的列
    available_cols = [c for c in cols_map.keys() if c in df_roster.columns]
    df_roster = df_roster[available_cols]

    # 重命名
    df_roster.rename(columns=cols_map, inplace=True)

    # --- 4. 排序 ---
    # 按 年份 -> 球队 -> 球员 排序
    df_roster = df_roster.sort_values(by=['Year', 'Tm', 'Player'])

    # --- 5. 保存 ---
    output_file = 'NBA_Rosters_2002_2026.csv'
    df_roster.to_csv(output_file, index=False)

    print("-" * 30)
    print(f"处理成功！")
    print(f"数据范围: {df_roster['Year'].min()} - {df_roster['Year'].max()}")
    print(f"总记录数: {len(df_roster)}")
    print(f"结果已保存为: {output_file}")

    # 预览
    print("\n数据预览 (2024赛季 示例):")
    print(df_roster[df_roster['Year'] == 2024].head())


# --- 执行部分 ---

if __name__ == "__main__":
    # 根据你的截图，这是你的文件相对路径
    # 如果运行报错，请右键点击左侧的 'Player Per Game.csv' -> 'Copy Path/Reference' -> 'Absolute Path'
    # 然后粘贴到下面：

    csv_path = r'kagglehub_new/datasets/sumitrodatta/nba-aba-baa-stats/versions/50/Player Per Game.csv'

    # 为了防止路径错误，我加了一个简单的自动修正（如果你把文件放在了同级目录）
    if not os.path.exists(csv_path):
        if os.path.exists('Player Per Game.csv'):
            csv_path = 'Player Per Game.csv'

    process_nba_rosters(csv_path)