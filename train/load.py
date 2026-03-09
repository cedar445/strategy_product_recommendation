import torch
import pandas as pd
import torch.nn as nn
from torch_geometric.nn.models.tgn import TGNMemory, LastNeighborLoader
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from torch_geometric.nn import TransformerConv

# =========================
# 1. 基础配置（必须与训练时一致）
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./model/tgn_model.pth"
MAP_DIR = "../data/trainData/csv/tgnCsv/"

MEMORY_DIM = 128
TIME_DIM = 128
EMBED_DIM = 128

# 加载映射表（用于 ID 转换）
product_map = pd.read_csv(MAP_DIR + "product_id_map.csv").set_index("original_id")["index"].to_dict()
strategy_map = pd.read_csv(MAP_DIR + "strategy_type_map.csv").set_index("original_id")["index"].to_dict()
# 反向映射：从索引找回策略名称
inv_strategy_map = {v: k for k, v in strategy_map.items()}

num_products = len(product_map)
num_strategies = len(strategy_map)
num_nodes = num_products + num_strategies

# =========================
# 2. 初始化模型架构
# =========================
memory = TGNMemory(
    num_nodes=num_nodes,
    raw_msg_dim=1,
    memory_dim=MEMORY_DIM,
    time_dim=TIME_DIM,
    message_module=IdentityMessage(1, MEMORY_DIM, TIME_DIM),
    aggregator_module=LastAggregator()
).to(DEVICE)

gnn = TransformerConv(MEMORY_DIM, EMBED_DIM, heads=2, concat=False).to(DEVICE)


class LinkPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim * 2, dim)
        self.lin2 = nn.Linear(dim, 1)

    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim=-1)
        h = torch.relu(self.lin1(h))
        return torch.sigmoid(self.lin2(h))


link_pred = LinkPredictor(EMBED_DIM).to(DEVICE)

# =========================
# 3. 加载权重
# =========================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
memory.load_state_dict(checkpoint["memory"])
gnn.load_state_dict(checkpoint["gnn"])
link_pred.load_state_dict(checkpoint["link_pred"])

memory.eval()
gnn.eval()
link_pred.eval()

# 邻居加载器（推理时也需要，用于聚合特征）
neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=DEVICE)


# =========================
# 4. 推理函数：预测商品的最优策略
# =========================

def recommend_strategy(original_product_id, top_k=1):
    """
    输入原始商品ID，输出模型认为最合适的策略
    """
    if original_product_id not in product_map:
        return "商品 ID 不存在"

    u_idx = product_map[original_product_id]

    # 所有的策略节点索引
    # 注意：策略索引在训练时加了 num_products 的偏移量
    strategy_indices = [v for v in strategy_map.values()]
    s_nodes = torch.tensor(strategy_indices, dtype=torch.long, device=DEVICE)
    u_node = torch.tensor([u_idx], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        # 1. 找到涉及到的所有节点并获取邻居
        n_id = torch.cat([u_node, s_nodes]).unique()
        n_id_gnn, edge_index, _ = neighbor_loader(n_id)

        # 2. 获取 Embedding
        z, _ = memory(n_id_gnn)
        z = gnn(z, edge_index)

        # 3. 映射回局部索引
        assoc = torch.empty(num_nodes, dtype=torch.long, device=DEVICE)
        assoc[n_id_gnn] = torch.arange(n_id_gnn.size(0), device=DEVICE)

        u_embed = z[assoc[u_node]].repeat(len(s_nodes), 1)  # 复制商品向量以匹配多个策略
        s_embeds = z[assoc[s_nodes]]

        # 4. 计算分数
        scores = link_pred(u_embed, s_embeds).squeeze()

        # 5. 排序并返回结果
        top_scores, top_indices = torch.topk(scores, k=top_k)

        results = []
        if top_k == 1:
            best_idx = s_nodes[top_indices].item()
            return inv_strategy_map[best_idx]
        else:
            for i in range(top_k):
                idx = s_nodes[top_indices[i]].item()
                results.append((inv_strategy_map[idx], top_scores[i].item()))
            return results


# =========================
# 测试运行
# =========================
if __name__ == "__main__":
    test_pid = "JI3215"
    result = recommend_strategy(test_pid, top_k=3)
    print(f"商品 {test_pid} 的推荐策略排名：")
    for res in result:
        print(f"策略: {res[0]}, 置信度: {res[1]:.4f}")