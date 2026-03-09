import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import os
import json
# ================= CONFIG =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "156278Lsk"
NEO4J_DATABASE = "weekly01"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15


def export_tgn_data():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    query = """
    MATCH (s:Store)<-[:AT_STORE]-(pst:PST)-[:OF_PRODUCT]->(p:Product)
    MATCH (pst)-[:IN_WEEK]->(w:Week)
    OPTIONAL MATCH (pst)-[:STRATEGY_ASSIGNED]->(leaf:StrategyLeaf)
    RETURN 
        elementId(s) AS u, 
        elementId(p) AS i, 
        pst.week_index AS ts, 
        leaf.code AS label,
        pst.sales_qty AS f1, 
        pst.inv_qty AS f2, 
        pst.discount_rate AS f3,
        pst.sales_amt_actual AS f4  // 将缺失的排名替换为实打实的销售额
    ORDER BY ts ASC
    """

    print("正在从 Neo4j 抽取数据...")
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)
        df = pd.DataFrame([dict(record) for record in result])
    driver.close()

    # --- 关键预处理 ---
    # 1. 跨年处理：将离散周索引映射为连续整数 (0, 1, 2...)
    unique_weeks = sorted(df['ts'].unique())
    week_map = {week: i for i, week in enumerate(unique_weeks)}
    df['ts'] = df['ts'].map(week_map)

    # 2. 节点 ID 防碰撞 (Product ID 偏移)
    # 对节点进行重新编码（将 elementId 映射为 0, 1, 2...）
    # 这一步会自动处理字符串 ID 无法相加的问题，且保证 ID 连续
    nodes = pd.concat([df['u'], df['i']]).unique()
    node_map = {node: i for i, node in enumerate(nodes)}

    df['u'] = df['u'].map(node_map)
    df['i'] = df['i'].map(node_map)

    # 3. 标签与特征填充
    df['label'] = df['label'].fillna('NO_ACT')
    df['label'], _ = pd.factorize(df['label'])
    df[['f1', 'f2', 'f3', 'f4']] = df[['f1', 'f2', 'f3', 'f4']].fillna(0)

    # --- 时间顺序划分 ---
    # --- 增强版时间划分 ---
    unique_ts = sorted(df['ts'].unique())
    n = len(unique_ts)

    # 强制逻辑：至少留出 1 个周给验证集，1 个周给测试集
    # 如果数据太少 (<= 3周)，则不划分测试集，防止报错
    if n <= 3:
        train_df = df
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
        print("警告：数据量过小，无法进行有效的训练/验证/测试划分！")
    else:
        # 动态计算切分点
        val_idx = max(int(n * TRAIN_SPLIT), n - 2)  # 保证至少留2个时间步给后续
        test_idx = max(int(n * (TRAIN_SPLIT + VAL_SPLIT)), n - 1)  # 保证至少留1个时间步

        train_cutoff = unique_ts[val_idx - 1]
        val_cutoff = unique_ts[test_idx - 1]

        train_df = df[df['ts'] <= train_cutoff]
        val_df = df[(df['ts'] > train_cutoff) & (df['ts'] <= val_cutoff)]
        test_df = df[df['ts'] > val_cutoff]

        train_df.to_csv("trainData/csv/tgnCsv/train.csv", index=False)
        val_df.to_csv("trainData/csv/tgnCsv/val.csv", index=False)
        test_df.to_csv("trainData/csv/tgnCsv/test.csv", index=False)

    print(f"导出完成！训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

    # 在保存之前，将字典的 key 强制转换为 str（或者将 dict 整体转为原生 Python 类型）
    # 使用 str(k) 将所有的键转换为字符串，这是最稳妥的 JSON 序列化方式
    def convert_to_json_friendly(d):
        return {str(k): v for k, v in d.items()}

    # 替换你原有的 json.dump 部分：
    with open("trainData/csv/tgnCsv/node_map.json", "w") as f:
        json.dump(convert_to_json_friendly(node_map), f)
    with open("trainData/csv/tgnCsv/week_map.json", "w") as f:
        json.dump(convert_to_json_friendly(week_map), f)


if __name__ == "__main__":
    export_tgn_data()