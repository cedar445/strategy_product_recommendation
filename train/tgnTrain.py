import pandas as pd
import torch.nn.functional as F
import torch
from torch.nn import MultiheadAttention, Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.data import TemporalData
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader  # 你刚才发现它在这里！
)

# ================= CONFIG =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "trainData/csv/tgnCsv/"
BATCH_SIZE = 200
LR = 0.001
EPOCHS = 50


# ================= 1. 加载数据 =================
def load_data(file_name):
    df = pd.read_csv(DATA_PATH + file_name)
    # 提取 TGN 所需的张量
    src = torch.tensor(df['u'].values, dtype=torch.long)
    dst = torch.tensor(df['i'].values, dtype=torch.long)
    t = torch.tensor(df['ts'].values, dtype=torch.long)
    msg = torch.tensor(df[['f1', 'f2', 'f3', 'f4']].values, dtype=torch.float)
    y = torch.tensor(df['label'].values, dtype=torch.long)
    return TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)


train_data = load_data("train.csv").to(DEVICE)
val_data = load_data("val.csv").to(DEVICE)
test_data = load_data("test.csv").to(DEVICE)

# 自动获取节点总数和类别总数
num_nodes = max(train_data.src.max(), train_data.dst.max(),
                val_data.src.max(), val_data.dst.max()) + 1
num_classes = len(torch.unique(train_data.y))


# ================= 2. 定义模型结构 =================
class TGNRecommender(torch.nn.Module):
    def __init__(self, num_nodes, msg_dim, memory_dim, time_dim, out_channels):
        super().__init__()

        # 记忆模块
        self.memory = TGNMemory(
            num_nodes, msg_dim, memory_dim, time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        # 替代 TemporalAttentionAggregation：使用标准的多头注意力
        # 注意：TGN 聚合后的维度依然是 memory_dim
        self.mha = MultiheadAttention(embed_dim=memory_dim, num_heads=2, batch_first=True)

        # 分类器
        self.classifier = Sequential(
            Linear(memory_dim, 64),
            ReLU(),
            BatchNorm1d(64),
            Linear(64, out_channels)
        )

    def forward(self, n_id):
        # 获取当前节点的记忆嵌入
        h, last_update = self.memory(n_id)

        # 模拟聚合过程：在实际复杂场景中，这里会根据邻居进行注意力计算
        # 这里我们直接将记忆送入分类器（这是 TGN 的基础版逻辑）
        return self.classifier(h)


# 初始化模型
model = TGNRecommender(
    num_nodes=int(num_nodes),
    msg_dim=4,
    memory_dim=64,
    time_dim=32,
    out_channels=num_classes
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# 辅助工具：用于获取历史邻居
# neighbor_loader = LastNeighborLoader(int(num_nodes), size=10).to(DEVICE)
# 既然 LastNeighborLoader 在这里，就这样初始化：
neighbor_loader = LastNeighborLoader(num_nodes, size=10)

# ================= 3. 训练与测试函数 =================
def train():
    model.train()
    model.memory.reset_state()  # 每个 epoch 开始前重置记忆
    neighbor_loader.reset_state()

    total_loss = 0
    # TGN 必须按时间顺序分批
    for batch in train_data.to_sequential_batches(BATCH_SIZE):
        optimizer.zero_grad()

        # 更新记忆并获取当前嵌入
        # n_id 包含了当前 batch 涉及的所有节点
        n_id = torch.cat([batch.src, batch.dst]).unique()
        h, _ = model.memory(n_id)

        # 我们预测目标节点 (Product) 的策略
        # 在 TGN 逻辑中，通常取 dst 节点的嵌入来做分类
        out = model.classifier(h[batch.dst])

        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        # 关键：更新 Memory (基于当前 batch 的交互)
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(data):
    model.eval()
    # 注意：测试时不需要重置记忆，因为它需要接续训练集的末尾状态
    total_correct = 0
    for batch in data.to_sequential_batches(BATCH_SIZE):
        n_id = torch.cat([batch.src, batch.dst]).unique()
        h, _ = model.memory(n_id)
        out = model.classifier(h[batch.dst])
        total_correct += int((out.argmax(dim=-1) == batch.y).sum())
        # 测试时也要更新记忆，以保证时序连续
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
    return total_correct / data.num_events


# ================= 4. 主循环 =================
print(f"开始训练，设备: {DEVICE}...")
for epoch in range(1, EPOCHS + 1):
    loss = train()
    val_acc = test(val_data)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

test_acc = test(test_data)
print(f'最终测试集准确率: {test_acc:.4f}')

# 保存模型
torch.save(model.state_dict(), "tgn_strategy_model.pt")
print("模型已保存为 tgn_strategy_model.pt")