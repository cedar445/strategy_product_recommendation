# ============================================================
# Weekly Parquet → Neo4j Importer (Advanced Graph Version)
# ============================================================

import os
import pyarrow.parquet as pq
from neo4j import GraphDatabase
from tqdm import tqdm

# ================= CONFIG =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "156278Lsk"
NEO4J_DATABASE = "weekly02"
ROOT_DIR = "trainData/weeklyPart1"
BATCH_SIZE = 2000

# ==========================================
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ============================================================
# Create Constraints & Index
# ============================================================

def create_constraints():
    with driver.session(database=NEO4J_DATABASE) as session:
        # 基础节点约束
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE (p.tenant, p.product_id) IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Store) REQUIRE (s.tenant, s.store_id) IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (w:Week) REQUIRE w.week_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (pst:PST) REQUIRE (pst.tenant, pst.product_id, pst.store_id, pst.week_id) IS UNIQUE")

        # 策略树节点约束 (新增)
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:StrategyRoot) REQUIRE r.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:StrategyBranch) REQUIRE b.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:StrategyLeaf) REQUIRE l.code IS UNIQUE")

        # 索引
        session.run("CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.prod_category, p.prod_brand)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (pst:PST) ON (pst.tenant, pst.product_id, pst.store_id, pst.week_index)")

        # 初始化策略树全景
        session.run("""
        // ============================================================
        // 1. 创建所有 策略根节点 (StrategyRoot)
        // ============================================================
        MERGE (r1:StrategyRoot {id: 'SR_BASKET'})   SET r1.name = '连带策略'
        MERGE (r2:StrategyRoot {id: 'SR_VIP'})      SET r2.name = '会员策略'
        MERGE (r3:StrategyRoot {id: 'SR_RANK'})     SET r3.name = '排名策略'
        MERGE (r4:StrategyRoot {id: 'SR_PRICING'})  SET r4.name = '定价策略'
        MERGE (r5:StrategyRoot {id: 'SR_INV'})      SET r5.name = '库存策略'
        MERGE (r6:StrategyRoot {id: 'SR_CLIMATE'})  SET r6.name = '气候策略'
        
        // ============================================================
        // 2. 连带策略树 (Basket Strategy)
        // ============================================================
        MERGE (b1:StrategyBranch {id: 'SB_MULTI'})  SET b1.name = '多件折扣'
        MERGE (b2:StrategyBranch {id: 'SB_BUNDLE'}) SET b2.name = '套装搭配'
        MERGE (r1)-[:HAS_BRANCH]->(b1)
        MERGE (r1)-[:HAS_BRANCH]->(b2)
        
        MERGE (l1:StrategyLeaf {code: 'ACT_MB_01'}) SET l1.name = '第二件9折'
        MERGE (l2:StrategyLeaf {code: 'ACT_MB_02'}) SET l2.name = '主题固定价'
        MERGE (b1)-[:HAS_LEAF]->(l1)
        MERGE (b2)-[:HAS_LEAF]->(l2)
        
        // ============================================================
        // 3. 会员策略树 (VIP Strategy)
        // ============================================================
        MERGE (b3:StrategyBranch {id: 'SB_REPURCHASE'}) SET b3.name = '复购唤醒'
        MERGE (r2)-[:HAS_BRANCH]->(b3)
        
        MERGE (l3:StrategyLeaf {code: 'ACT_VIP_01'}) SET l3.name = '限时复购券'
        MERGE (b3)-[:HAS_LEAF]->(l3)
        
        // ============================================================
        // 4. 排名策略树 (Ranking Strategy)
        // ============================================================
        MERGE (b4:StrategyBranch {id: 'SB_TOP_RANK'}) SET b4.name = '标杆维持'
        MERGE (r3)-[:HAS_BRANCH]->(b4)
        
        MERGE (l4:StrategyLeaf {code: 'ACT_RNK_01'}) SET l4.name = '橱窗展示位'
        MERGE (b4)-[:HAS_LEAF]->(l4)
        
        // ============================================================
        // 5. 定价策略树 (Pricing Strategy)
        // ============================================================
        MERGE (b5:StrategyBranch {id: 'SB_DISCOUNT'}) SET b5.name = '折扣出清'
        MERGE (r4)-[:HAS_BRANCH]->(b5)
        
        MERGE (l5:StrategyLeaf {code: 'ACT_PRC_DEEP'}) SET l5.name = '3折清仓'
        MERGE (l6:StrategyLeaf {code: 'ACT_PRC_MID'})  SET l6.name = '7折促销'
        MERGE (b5)-[:HAS_LEAF]->(l5)
        MERGE (b5)-[:HAS_LEAF]->(l6)
        
        // ============================================================
        // 6. 库存策略树 (Inventory Strategy)
        // ============================================================
        MERGE (b6:StrategyBranch {id: 'SB_STOCK_ADJ'}) SET b6.name = '库存平衡'
        MERGE (r5)-[:HAS_BRANCH]->(b6)
        
        MERGE (l7:StrategyLeaf {code: 'ACT_INV_REPLENISH'}) SET l7.name = '黄金尺码补货'
        MERGE (l8:StrategyLeaf {code: 'ACT_INV_TRANSFER'})  SET l8.name = '库存调拨'
        MERGE (b6)-[:HAS_LEAF]->(l7)
        MERGE (b6)-[:HAS_LEAF]->(l8)
        
        // ============================================================
        // 7. 气候策略树 (Climate Strategy)
        // ============================================================
        MERGE (b7:StrategyBranch {id: 'SB_WEATHER'}) SET b7.name = '气候感应'
        MERGE (r6)-[:HAS_BRANCH]->(b7)
        
        MERGE (l9:StrategyLeaf {code: 'ACT_CLI_RAIN'}) SET l9.name = '雨天营销'
        MERGE (l10:StrategyLeaf {code: 'ACT_CLI_HEAT'}) SET l10.name = '高温推新'
        MERGE (b7)-[:HAS_LEAF]->(l9)
        MERGE (b7)-[:HAS_LEAF]->(l10)
        """)

# ============================================================
# Batch Write
# ============================================================

def batch_write(query, rows):
    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(lambda tx: tx.run(query, rows=rows))

# ============================================================
# Import Product + PST + Strategy
# ============================================================

def import_product_parquet(file_path):

    df = pq.read_table(file_path).to_pandas()

    df = df.fillna({
        "discount_rate": 1.0,  # 默认不打折
        "inv_qty": 0,
        "sales_amt_actual": 0,
        "sales_qty": 0,
        # "upt": 1.0,  # 默认连带率为1
        "total_retail_qty_for_upt": 0,
        "total_retail_cnt": 0,
        "is_in_short": 0,  # 默认不缺货
        "gold_full_size_rate": 1.0,  # 默认尺码齐全
        "inv_sale_rate_num": 0,  # 默认库销比为0
        "sales_amt_actual_store_rank": 999,  # 默认排名靠后
        "vip_retail_pct": 0,  # 默认无会员销售
        "prod_theme": ""  # 默认无主题
    })

    rows = df.to_dict("records")

    query = """
    UNWIND $rows AS row

    MERGE (p:Product { tenant: row.tenant, product_id: row.product_id })
    SET p.prod_name = row.prod_name,
        p.prod_brand = row.prod_brand,
        p.prod_category = row.prod_catagory,
        p.gender = row.gender,
        p.prod_season = row.prod_season

    MERGE (s:Store { tenant: row.tenant, store_id: row.store_id })
    MERGE (w:Week { week_id: row.week_id })

    MERGE (pst:PST {
        tenant: row.tenant,
        product_id: row.product_id,
        store_id: row.store_id,
        week_id: row.week_id
    })
    SET pst.sales_amt_actual = row.sales_amt_actual,
        pst.sales_qty = row.sales_qty,
        pst.discount_rate = row.discount_rate,
        pst.inv_qty = row.inv_qty,
        pst.life_cycle_stage = row.life_cycle_stage,
        pst.week_index =
            toInteger(substring(row.week_id,0,4)) * 100 +
            toInteger(substring(row.week_id,5,2))

    MERGE (pst)-[:OF_PRODUCT]->(p)
    MERGE (pst)-[:AT_STORE]->(s)
    MERGE (pst)-[:IN_WEEK]->(w)


    // --- 全维度策略判定逻辑 (构建策略集) ---
    WITH pst, row, [] AS codes

    // 1. 定价策略维度 (Pricing)
    WITH pst, row, codes + [CASE 
        WHEN row.discount_rate < 0.4 THEN "ACT_PRC_DEEP"   // 4折以下
        WHEN row.discount_rate < 0.8 THEN "ACT_PRC_MID"    // 4-8折
        ELSE null END] AS codes
        
    // 2. 连带策略维度 (Basket)
    WITH pst, row, codes + [CASE 
        WHEN row.total_retail_cnt > 0 
            AND (row.total_retail_qty_for_upt / row.total_retail_cnt) < 1.2 
            THEN "ACT_MB_01"               // 连带低，推第二件折扣
        WHEN row.prod_theme <> "" THEN "ACT_MB_02"        // 有主题，推套装搭配
        ELSE null END] AS codes

    // 3. 库存策略维度 (Inventory)
    WITH pst, row, codes + [CASE 
        WHEN row.is_in_short = 1 OR row.gold_full_size_rate < 0.6 THEN "ACT_INV_REPLENISH" // 缺货或断码
        WHEN row.inv_sale_rate_num > 5.0 AND row.sales_qty < 2 THEN "ACT_INV_TRANSFER"     // 积压且不动销
        ELSE null END] AS codes
        
    // 4. 排名策略维度 (Ranking)
    WITH pst, row, codes + [CASE 
        WHEN row.sales_amt_actual_store_rank <= 10 THEN "ACT_RNK_01" // 店内前10名，给橱窗位
        ELSE null END] AS codes

    // 5. 会员策略维度 (VIP)
    WITH pst, row, codes + [CASE 
        WHEN row.vip_retail_pct > 0.4 THEN "ACT_VIP_01"   // 会员贡献高，发复购券
        ELSE null END] AS codes

    // --- C. 批量执行连线 (二部图构建) ---
    UNWIND [c IN codes WHERE c IS NOT NULL] AS leaf_code
    MATCH (leaf:StrategyLeaf {code: leaf_code})
    MERGE (pst)-[:APPLY]->(leaf)
    """

    for i in tqdm(range(0, len(rows), BATCH_SIZE)):
        batch_write(query, rows[i:i+BATCH_SIZE])

# ============================================================
# Temporal Links -- EFFECT OF
# ============================================================
def create_temporal_links():

    query_clear = """
    MATCH (n:PST:Processed)
    REMOVE n:Processed
    """

    query = """
    MATCH (pst1:PST)
    WHERE NOT pst1:Processed
    WITH pst1 LIMIT 2000

    SET pst1:Processed
    WITH pst1  // <--- 这里是关键：添加这行以保持上下文

    OPTIONAL MATCH (pst1)-[:APPLY]->(leaf:StrategyLeaf)
    WITH pst1, collect(leaf.code) AS strategy_list

    MATCH (pst2:PST {
        tenant: pst1.tenant,
        product_id: pst1.product_id,
        store_id: pst1.store_id
    })
    WHERE pst2.week_index = pst1.week_index + 1
       OR pst2.week_index = (toInteger(floor(pst1.week_index/100))+1)*100+1

    MERGE (pst1)-[e:EFFECT_OF]->(pst2)
    SET e.total_sales_lift = pst2.sales_qty - pst1.sales_qty,
        e.total_amt_lift = pst2.sales_amt_actual - pst1.sales_amt_actual,
        e.strategies = strategy_list,
        e.strategy_count = size(strategy_list)
    """

    with driver.session(database=NEO4J_DATABASE) as session:

        session.run(query_clear)

        while True:
            result = session.run(query)
            summary = result.consume()

            if summary.counters.relationships_created == 0:
                remaining = session.run("""
                MATCH (p:PST)
                WHERE NOT p:Processed
                RETURN count(p) AS c
                """).single()["c"]

                if remaining == 0:
                    session.run("MATCH (n:PST:Processed) REMOVE n:Processed")
                    break


# ============================================================
# Main
# ============================================================

def main():

    create_constraints()

    for week_folder in sorted(os.listdir(ROOT_DIR)):

        week_path = os.path.join(ROOT_DIR, week_folder)

        if not os.path.isdir(week_path):
            continue

        product_file = os.path.join(week_path, "product.parquet")

        if os.path.exists(product_file):
            print(f"Importing {week_folder}")
            import_product_parquet(product_file)

    print("Creating temporal links...")
    create_temporal_links()




if __name__ == "__main__":
    main()
    driver.close()