import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import webbrowser


def create_static_highlight_visible(edges_file='nba_active_player_edges.csv',
                                    community_file='nba_player_communities.csv'):
    print("1. 正在读取数据...")
    if not os.path.exists(edges_file) or not os.path.exists(community_file):
        print("错误：找不到文件，请确认路径。")
        return

    edges = pd.read_csv(edges_file)
    coms = pd.read_csv(community_file)

    # --- 2. 计算布局 ---
    print("2. 计算静态布局...")
    G = nx.from_pandas_edgelist(edges, 'Source', 'Target', edge_attr='Weight')
    edges_to_layout = [(u, v) for u, v, d in G.edges(data=True) if d['Weight'] >= 2]
    G_layout = G.edge_subgraph(edges_to_layout).copy()

    for node in coms['Player']:
        if node not in G_layout: G_layout.add_node(node)

    pos = nx.spring_layout(G_layout, k=0.18, seed=42, iterations=100)

    # --- 3. PyVis 设置 ---
    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="black", select_menu=False)

    # 锁定配置
    options_str = """
    var options = {
      "nodes": { "font": { "size": 0 } },
      "interaction": {
        "dragNodes": false, 
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "multiselect": false
      },
      "physics": { "enabled": false }
    }
    """
    net.set_options(options_str)

    # 颜色生成
    unique_groups = sorted(coms['Group_ID'].unique())
    cmap = plt.get_cmap('tab20', len(unique_groups))
    hex_colors = {}
    for i, gid in enumerate(unique_groups):
        hex_colors[gid] = mcolors.to_hex(cmap(i))

    print("3. 绘制节点...")
    SCALE_FACTOR = 2500

    for index, row in coms.iterrows():
        player = row['Player']
        gid = row['Group_ID']
        role = row['Role']

        if player in pos:
            x_pos = pos[player][0] * SCALE_FACTOR
            y_pos = pos[player][1] * SCALE_FACTOR
        else:
            x_pos, y_pos = 0, 0

        color = hex_colors.get(gid, '#cccccc')
        size = 45 if role == 'Core' else 15

        net.add_node(player,
                     x=x_pos, y=y_pos,
                     label=" ",
                     title=f"<b>{player}</b><br>Group: {gid}<br>Role: {role}",
                     color=color,
                     size=size,
                     group=gid)

    print("4. 绘制连线...")
    for index, row in edges.iterrows():
        w = row['Weight']
        if w >= 2:
            net.add_edge(row['Source'], row['Target'],
                         value=w, width=w * 0.3,
                         color={'color': '#cccccc', 'opacity': 0.6})

    # --- 5. 注入调整后的交互脚本 ---
    output_file = "index.html"
    print(f"5. 生成网页: {output_file} ...")

    html_content = net.generate_html()

    custom_js = """
    <script type="text/javascript">
        var isInitialized = false;

        network.on("click", function (params) {
            var allNodes = nodes.get();
            var allEdges = edges.get();

            // 初始化保存原始颜色
            if (!isInitialized) {
                var nodeUpdates = [];
                var edgeUpdates = [];
                for (var i = 0; i < allNodes.length; i++) {
                    var n = allNodes[i];
                    n.orgColor = n.color;
                    nodeUpdates.push(n);
                }
                for (var i = 0; i < allEdges.length; i++) {
                    var e = allEdges[i];
                    if (typeof e.color === 'object' && e.color !== null) {
                       e.baseColorString = e.color.color || '#cccccc';
                    } else {
                       e.baseColorString = e.color || '#cccccc';
                    }
                    edgeUpdates.push(e);
                }
                nodes.update(nodeUpdates);
                edges.update(edgeUpdates);
                isInitialized = true;
            }

            if (params.nodes.length > 0) {
                // === 选中状态 ===
                var selectedNodeId = params.nodes[0];
                var selectedNode = nodes.get(selectedNodeId);
                var targetGroup = selectedNode.group;

                var nodeUpdates = [];
                var edgeUpdates = [];

                // 1. 更新节点
                for (var i = 0; i < allNodes.length; i++) {
                    var node = allNodes[i];
                    if (node.group == targetGroup) {
                        // 同组：高亮
                        nodeUpdates.push({
                            id: node.id,
                            color: node.orgColor,
                            opacity: 1.0
                        });
                    } else {
                        // 异组：调整这里的参数来控制"背景板"的清晰度
                        // 修改：颜色加深到 #d0d0d0 (浅灰)，透明度提升到 0.4
                        nodeUpdates.push({
                            id: node.id,
                            color: { background: '#d0d0d0', border: '#c0c0c0' },
                            opacity: 0.4 
                        });
                    }
                }

                // 2. 更新连线
                for (var i = 0; i < allEdges.length; i++) {
                    var edge = allEdges[i];
                    var fromNode = nodes.get(edge.from);
                    var toNode = nodes.get(edge.to);

                    if (fromNode.group == targetGroup && toNode.group == targetGroup) {
                        edgeUpdates.push({
                            id: edge.id,
                            color: { color: edge.baseColorString, opacity: 0.8 },
                            width: edge.width
                        });
                    } else {
                        // 异组连线：保持微弱可见，不完全消失
                        edgeUpdates.push({
                            id: edge.id,
                            color: { color: '#e0e0e0', opacity: 0.1 },
                            width: 0.1 
                        });
                    }
                }

                nodes.update(nodeUpdates);
                edges.update(edgeUpdates);

            } else {
                // === 还原状态 ===
                var nodeUpdates = [];
                var edgeUpdates = [];
                for (var i = 0; i < allNodes.length; i++) {
                    var node = allNodes[i];
                    if (node.orgColor) {
                        nodeUpdates.push({
                            id: node.id,
                            color: node.orgColor,
                            opacity: 1.0
                        });
                    }
                }
                for (var i = 0; i < allEdges.length; i++) {
                    var edge = allEdges[i];
                    if (edge.baseColorString) {
                         edgeUpdates.push({
                            id: edge.id,
                            color: { color: edge.baseColorString, opacity: 0.5 },
                            width: edge.width
                        });
                    }
                }
                nodes.update(nodeUpdates);
                edges.update(edgeUpdates);
            }
        });
    </script>
    """

    if "</body>" in html_content:
        html_content = html_content.replace("</body>", custom_js + "\n</body>")
    else:
        html_content += custom_js

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print("成功！正在打开浏览器...")
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
    except Exception as e:
        print(f"保存出错: {e}")


if __name__ == "__main__":
    create_static_highlight_visible()