import pandas as pd
import numpy as np
import csv
from neo4j import GraphDatabase

# 定义所有可能的策略（确保与你的 StrategyLeaf 一致）
ALL_STRATEGIES = [
    'ACT_PRC_DEEP', 'ACT_PRC_MID', 'ACT_MB_01', 'ACT_MB_02',
    'ACT_VIP_01', 'ACT_RNK_01', 'ACT_INV_REPLENISH',
    'ACT_INV_TRANSFER', 'ACT_CLI_RAIN', 'ACT_CLI_HEAT'
]

output_path="trainData/csv/tgnCsv/"

def export_and_process_data(driver, output_base=output_path):
    # 1. 导出图数据
    query = """
    MATCH (p:Product)-[:OF_PRODUCT]-(pst1:PST)-[e:EFFECT_OF]->(pst2:PST)
    MATCH (pst1)-[:AT_STORE]->(s:Store)
    RETURN p.product_id AS src, 
           s.store_id AS dst, 
           pst1.week_index AS time, 
           e.strategies AS strategy,
           pst1.sales_qty AS sales_prev,
           pst2.sales_qty AS sales_next
    """
    with driver.session(database="weekly02") as session:
        results = session.run(query)
        df = pd.DataFrame([dict(record) for record in results])

    # 2. 特征工程 (One-Hot 编码)
    print("开始进行特征编码...")
    for s_code in ALL_STRATEGIES:
        df[f'feat_{s_code}'] = df['strategy'].apply(lambda x: 1 if s_code in x else 0)

    # 计算目标标签 (Sales Lift)
    df['label'] = df['sales_next'] - df['sales_prev']

    # 3. 按时间划分数据集
    print("开始进行按时间划分...")
    df = df.sort_values(by='time')

    total_len = len(df)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # 4. 保存文件
    train_df.to_csv(f"{output_base}/train.csv", index=False)
    val_df.to_csv(f"{output_base}/val.csv", index=False)
    test_df.to_csv(f"{output_base}/test.csv", index=False)

    print(f"数据已导出: 训练集({len(train_df)}), 验证集({len(val_df)}), 测试集({len(test_df)})")

#