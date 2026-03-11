import pandas as pd
import torch
import os
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, MultiheadAttention
from torch_geometric.data import TemporalData
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader
)

# ================= CONFIG =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 建议此处使用绝对路径，避免找不到文件
DATA_PATH = "../data/trainData/csv/tgnCsv"
BATCH_SIZE = 200
LR = 0.0005
EPOCHS = 30


# ================= 1. 加载数据 =================
def load_data(file_name):
    path = os.path.join(DATA_PATH, file_name)
    if not os.path.exists(path):
        print(f"警告：找不到文件 {path}")
        return None

    df = pd.read_csv(path)
    if len(df) == 0:
        return None

    # 转换为 Tensor
    src = torch.tensor(df['u'].values, dtype=torch.long)
    dst = torch.tensor(df['i'].values, dtype=torch.long)
    t = torch.tensor(df['ts'].values, dtype=torch.long)
    msg = torch.tensor(df[['f1', 'f2', 'f3', 'f4']].values, dtype=torch.float)
    y = torch.tensor(df['label'].values, dtype=torch.long)

    return TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)


# 预加载所有数据以计算全局参数
train_data = load_data("train.csv")
val_data = load_data("val.csv")
test_data = load_data("test.csv")

if train_data is None:
    raise FileNotFoundError("无法加载训练数据，请检查路径和文件内容。")

# --- 核心修复：计算全局最大节点 ID 和类别数，防止索引越界 ---
all_src = [train_data.src]
all_dst = [train_data.dst]
all_y = [train_data.y]

if val_data:
    all_src.append(val_data.src)
    all_dst.append(val_data.dst)
    all_y.append(val_data.y)
if test_data:
    all_src.append(test_data.src)
    all_dst.append(test_data.dst)
    all_y.append(test_data.y)

num_nodes = int(torch.cat(all_src + all_dst).max() + 1)
num_classes = int(torch.cat(all_y).max() + 1)

print(f"--- 初始化配置 ---")
print(f"运行设备: {DEVICE}")
print(f"全局节点总数 (num_nodes): {num_nodes}")
print(f"预测类别总数 (num_classes): {num_classes}")
print(f"------------------")

# 将数据移至设备
train_data = train_data.to(DEVICE)


# ================= 2. 定义模型 =================
class TGNRecommender(torch.nn.Module):
    def __init__(self, num_nodes, msg_dim, memory_dim, time_dim, out_channels):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes, msg_dim, memory_dim, time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        # 简单高效的分类头
        self.classifier = Sequential(
            Linear(memory_dim, 64),
            ReLU(),
            BatchNorm1d(64),
            Linear(64, out_channels)
        )

    def forward(self, n_id):
        h, _ = self.memory(n_id)
        return self.classifier(h)


model = TGNRecommender(
    num_nodes=num_nodes,
    msg_dim=4,
    memory_dim=100,
    time_dim=100,
    out_channels=num_classes
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()
neighbor_loader = LastNeighborLoader(num_nodes, size=10)  # 保持在 CPU


# ================= 3. 训练与评估逻辑 =================
def train():
    model.train()
    model.memory.reset_state()
    neighbor_loader.reset_state()

    total_loss = 0
    for i in range(0, train_data.num_events, BATCH_SIZE):
        batch = train_data[i: i + BATCH_SIZE].to(DEVICE)
        optimizer.zero_grad()

        # --- 核心修复：断开历史梯度链 ---
        model.memory.detach()

        n_id = torch.cat([batch.src, batch.dst]).unique()
        out = model(n_id)

        # 建立局部索引映射
        node_to_idx = {node.item(): i for i, node in enumerate(n_id)}
        batch_dst_idx = torch.tensor([node_to_idx[d.item()] for d in batch.dst], device=DEVICE)

        loss = criterion(out[batch_dst_idx], batch.y)
        loss.backward()
        optimizer.step()

        # 更新记忆与邻居（注意 .cpu() 转换）
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src.cpu(), batch.dst.cpu())

        total_loss += loss.item() * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def evaluate(data):
    if data is None: return 0.0
    model.eval()
    # 评估时保持记忆连续性
    # 不重置 memory 状态，因为测试是基于训练后的状态进行的
    total_correct = 0
    for i in range(0, data.num_events, BATCH_SIZE):
        batch = data[i: i + BATCH_SIZE].to(DEVICE)
        n_id = torch.cat([batch.src, batch.dst]).unique()

        # 同样需要 detach 防止推理时内存累积
        model.memory.detach()

        out = model(n_id)
        node_to_idx = {node.item(): i for i, node in enumerate(n_id)}
        batch_dst_idx = torch.tensor([node_to_idx[d.item()] for d in batch.dst], device=DEVICE)

        total_correct += int((out[batch_dst_idx].argmax(dim=-1) == batch.y).sum())
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)

    return total_correct / data.num_events


# ================= 4. 运行 =================
for epoch in range(1, EPOCHS + 1):
    loss = train()
    val_acc = evaluate(val_data)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

if test_data:
    test_acc = evaluate(test_data)
    print(f'最终测试集准确率: {test_acc:.4f}')

torch.save(model.state_dict(), "tgn_model.pt")
print("训练完成，模型已保存。")