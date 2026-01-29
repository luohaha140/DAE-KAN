import os
import torch
import numpy as np
from tqdm import tqdm

from model.model import model_KAN
from utils.LBFGS import LBFGS
from utils.utils import batch_jacobian

# 设置设备和数据类型
device = torch.device("cpu")
print("Using device:", device)
torch.set_default_tensor_type(torch.DoubleTensor)


def DAE_loss(y_n):
    """计算微分代数方程的总损失"""
    yn = y_n.clone().detach().requires_grad_(True)
    y_pred = model(yn)
    grads = batch_jacobian(model, yn, create_graph=True)

    # 提取预测变量
    u_1, u_2, z_1, z_2, v = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3], y_pred[:, 4]

    # 计算时间导数
    u_1_t = grads[:, 0, 0]
    u_2_t = grads[:, 1, 0]
    z_1_t = grads[:, 2, 0]
    z_2_t = grads[:, 3, 0]

    # DAE 方程残差
    f1 = u_1_t - z_1
    f2 = u_2_t - z_2
    f3 = 2 * u_2 - 2 * u_2 * u_2 * u_2 - u_1 * v - z_1_t
    f4 = 2 * u_1 - 2 * u_1 * u_1 * u_1 - u_2 * v - z_2_t
    f5 = u_1 * z_1 + u_2 * z_2

    # 方程损失
    F1 = torch.mean(f1 ** 2)
    F2 = torch.mean(f2 ** 2)
    F3 = torch.mean(f3 ** 2)
    F4 = torch.mean(f4 ** 2)
    F5 = torch.mean(f5 ** 2)
    F = F1 + F2 + F3 + F4 + F5

    # 初始条件
    x_0 = (yn[:, 0] == 0)
    y_0 = (yn[:, 0] == 0)
    u_0 = (yn[:, 0] == 0)
    v_0 = (yn[:, 0] == 0)
    d_0 = (yn[:, 0] == 0)

    # 初始条件损失
    x_init_loss = torch.mean((u_1[x_0] - 1) ** 2)
    y_init_loss = torch.mean((u_2[y_0] - 0) ** 2)
    u_init_loss = torch.mean((z_1[u_0] - 0) ** 2)
    v_init_loss = torch.mean((z_2[v_0] - 1) ** 2)
    d_init_loss = torch.mean((v[d_0] - 1) ** 2)
    init_loss = x_init_loss + y_init_loss + u_init_loss + v_init_loss + d_init_loss

    # 总损失
    total_loss = 1 * F1 + 1 * F2 + 1 * F3 + 1 * F4 + 1 * F5 + 2 * init_loss
    return total_loss


def pendulum_DAE(y_n):
    """为保持兼容性，调用 DAE_loss 函数"""
    return DAE_loss(y_n)


def exact_solution(t_sim):
    """计算精确解"""
    y1 = np.cos(t_sim)
    y2 = np.sin(t_sim)
    y3 = -np.sin(t_sim)
    y4 = np.cos(t_sim)
    y5 = 1 + np.sin(2 * t_sim)
    return np.array([y1, y2, y3, y4, y5])


def calculate_relative_L2_errors(y_pred_KAN, t):
    """计算相对 L2 误差"""
    y_exact = exact_solution(t).T
    relative_errors = np.zeros(5)

    for v in range(5):
        differences = y_pred_KAN[:, v] - y_exact[:, v]
        molecule = np.sqrt(np.sum(differences ** 2))
        denominator = np.sqrt(np.sum(y_exact[:, v] ** 2))
        relative_errors[v] = molecule / denominator

    return relative_errors


def train():
    """训练模型"""
    optimizer = LBFGS(
        model.parameters(),
        lr=1,
        history_size=80,
        tolerance_grad=1e-32,
        tolerance_change=1e-32,
        tolerance_ys=1e-32
    )

    steps = 200
    pbar = tqdm(range(steps), desc='Training Progress')

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = pendulum_DAE(t1)
            loss.backward()
            return loss

        closure()
        optimizer.step(closure)

        if step % 1 == 0:
            current_loss = closure().item()
            pbar.set_description("Step: %d | Loss: %.15f" % (step, current_loss))


# 主程序
if __name__ == "__main__":
    # 加载预训练模型
    m_state_dict = torch.load("../logs/model_particle_index2.pth")
    model = model_KAN(
        width_dyn=[1, 5, 5, 4],
        width_alg=[1, 5, 5, 1],
        grid_dyn=6,
        k_dyn=6,
        noise_dyn=0,
        grid_alg=6,
        k_alg=6,
        noise_alg=0
    ).double()
    model.load_state_dict(m_state_dict)

    # 创建时间网格
    t = np.arange(0.0, 1.005, 0.005)
    t1 = torch.Tensor(t)
    t1 = torch.unsqueeze(t1, dim=1)

    # 训练模型
    # train()

    # 保存模型
    # torch.save(model.state_dict(), "../logs/model_particle_index2.pth")

    # 计算预测和误差
    y_pred_KAN = model(t1).detach().numpy()
    l2_error_KAN = calculate_relative_L2_errors(y_pred_KAN, t)
    print("l2_errorKAN:", l2_error_KAN)