from config import *
from algorithms.traditional import ZFBeamformer
from models.beamforming import get_model_name, FairBeamformingNet, MultiTaskLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import torch.optim as optim
from models.channel_data_generate import generate_sensing_channel,generate_communication_channel

# ====================== Data generation functions ======================
def generate_training_data(num_samples=10000):
    X, Y = [], []
    num_users = len(user_angles)

    for _ in range(num_samples):
        main_idx = np.random.randint(0, num_users)
        main_user = user_angles[main_idx]
        other_users = [a for i, a in enumerate(user_angles) if i != main_idx]

        targets = [main_user] + other_users[:num_users - 1] + target_angles
        targets = targets[:input_size]

        ideal_weights = np.random.randn(num_antennas * 2)
        zf_processor = ZFBeamformer([main_user] + other_users[:num_users - 1])
        zf_weights = zf_processor.apply(ideal_weights[np.newaxis, :])[0]

        X.append(targets)
        # print(targets)
        Y.append(zf_weights)
        # print(zf_weights)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


# ====================== 数据生成 ======================
def generate_dataset(num_samples = 10000):

    Hc_real, Hc_imag = [], []
    Hs_real, Hs_imag = [], []
    rho_values = []

    for _ in tqdm(range(num_samples)):
        shuffled_user_angles = user_angles.copy()  # 避免修改原数组
        random.shuffle(shuffled_user_angles)  # 随机打乱顺序

        H_c = generate_communication_channel(num_antennas, shuffled_user_angles)
        # H_c = generate_communication_channel(num_antennas, user_angles)
        H_s = generate_sensing_channel(num_antennas, target_angles)
        # print('H_c:', H_c)
        # print('H_s:', H_s)

        Hc_real.append(torch.FloatTensor(H_c.real))
        Hc_imag.append(torch.FloatTensor(H_c.imag))
        Hs_real.append(torch.FloatTensor(H_s.real))
        Hs_imag.append(torch.FloatTensor(H_s.imag))
        # rho_values.append(torch.FloatTensor([np.random.beta(2, 2)]))
        rho_values.append(torch.FloatTensor([rho]))

    return (torch.stack(Hc_real), torch.stack(Hc_imag),
            torch.stack(Hs_real), torch.stack(Hs_imag),
            torch.stack(rho_values))


# ====================== 模型训练 ======================
def train_model(rho_set=None):
    # Generate standardized model names
    model_name = get_model_name(user_angles, target_angles, num_antennas, rho_set)
    # 生成数据集
    Hc_real, Hc_imag, Hs_real, Hs_imag, rho = generate_dataset(10000)  # 减少样本量便于测试

    # 划分训练验证集
    (Hc_real_train, Hc_real_val,
     Hc_imag_train, Hc_imag_val,
     Hs_real_train, Hs_real_val,
     Hs_imag_train, Hs_imag_val,
     rho_train, rho_val) = train_test_split(
        Hc_real, Hc_imag, Hs_real, Hs_imag, rho, test_size=0.2)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        Hc_real_train, Hc_imag_train,
        Hs_real_train, Hs_imag_train,
        rho_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)  # 减小batch size

    val_dataset = torch.utils.data.TensorDataset(
        Hc_real_val, Hc_imag_val,
        Hs_real_val, Hs_imag_val,
        rho_val
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64)

    # 模型参数
    # input_size = 2 * (num_users * num_antennas + num_targets * num_antennas) + 1
    model = FairBeamformingNet(input_size, hidden_size=512, max_users=num_users,num_rf_chains=num_rf_chains).to(device)  # 减小隐藏层大小
    criterion = MultiTaskLoss(rho=rho, lambda_reg=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(15):  # 减少epoch数量
        model.train()
        train_loss = 0
        for Hc_r, Hc_i, Hs_r, Hs_i, r in train_loader:
            # 移动数据到设备
            Hc_r, Hc_i = Hc_r.to(device), Hc_i.to(device)
            Hs_r, Hs_i = Hs_r.to(device), Hs_i.to(device)
            r = r.to(device)

            optimizer.zero_grad()

            # 前向传播 - 现在正确传递所有参数
            W = model(Hc_r, Hc_i, Hs_r, Hs_i, r)

            # 计算损失
            Hc = torch.complex(Hc_r, Hc_i)
            Hs = torch.complex(Hs_r, Hs_i)
            loss = criterion(W, Hc, Hs)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Hc_r, Hc_i, Hs_r, Hs_i, r in val_loader:
                Hc_r, Hc_i = Hc_r.to(device), Hc_i.to(device)
                Hs_r, Hs_i = Hs_r.to(device), Hs_i.to(device)
                r = r.to(device)

                W = model(Hc_r, Hc_i, Hs_r, Hs_i, r)
                Hc = torch.complex(Hc_r, Hc_i)
                Hs = torch.complex(Hs_r, Hs_i)
                val_loss += criterion(W, Hc, Hs).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_name)

    return model, model_name
