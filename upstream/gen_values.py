import numpy as np
import torch
from tqdm import tqdm

def generate_data_0(sample_num, n_agents, m_items, alpha, dx, dy, device):
    X = torch.rand((sample_num, n_agents, dx))
    X = X * 2 - 1
    Y = torch.rand((sample_num, m_items, dy))
    Y = Y * 2 - 1
    upper = np.fromfunction(lambda t, i, j: torch.sigmoid((X[t, i] * Y[t, j]).sum(-1)), (sample_num, n_agents, m_items)).clone().detach()
    values = torch.rand((sample_num, n_agents, m_items)) * upper
    return values.to(device), X.to(device), Y.to(device)

def generate_data_1(sample_num, n_agents, m_items, alpha, dx, dy, device): #只适用于item = 2
    X = torch.rand((sample_num, n_agents, dx)) * 2 - 1
    Y = torch.rand((sample_num, m_items, dy)) * 2 - 1
    upper = np.fromfunction(lambda t, i, j: torch.sigmoid((X[t, i] * Y[t, j]).sum(-1)), (sample_num, n_agents, m_items)).clone().detach() 
    values = torch.rand((sample_num, n_agents, 1))
    values = torch.cat((values, 1 - values), dim=-1)
    values = values * upper
    return values.to(device), X.to(device), Y.to(device)

def generate_data_2(sample_num, n_agents, m_items, alpha, dx, dy, device):
    values = torch.rand((sample_num, n_agents, m_items))
    X = torch.arange(n_agents).repeat(sample_num).reshape(sample_num, n_agents).long()
    Y = torch.arange(m_items).repeat(sample_num).reshape(sample_num, m_items).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_data_3(sample_num, n_agents, m_items, alpha, dx, dy, device):
    values = np.random.exponential(3.0, size=(sample_num, 3, 1))
    values = torch.tensor(values)
    X = torch.arange(3).repeat(sample_num).reshape(sample_num, 3).long()
    Y = torch.arange(1).repeat(sample_num).reshape(sample_num, 1).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_data_4(sample_num, n_agents, m_items, alpha, dx, dy, device):
    values1 = torch.rand((sample_num, 1, 1)) * 12 + 4
    values2 = torch.rand((sample_num, 1, 1)) * 3 + 4
    values = torch.cat((values1, values2), -1)
    X = torch.arange(1).repeat(sample_num).reshape(sample_num, 1).long()
    Y = torch.arange(2).repeat(sample_num).reshape(sample_num, 2).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_data_5(sample_num, n_agents, m_items, alpha, dx, dy, device):
    values1 = torch.rand((sample_num, 1, 1))
    values1 = (1 / (1 - values1)) ** 0.2 - 1
    values2 = torch.rand((sample_num, 1, 1))
    values2 = (1 / (1 - values2)) ** (1/6) - 1
    values = torch.cat((values1, values2), -1)
    X = torch.arange(1).repeat(sample_num).reshape(sample_num, 1).long()
    Y = torch.arange(2).repeat(sample_num).reshape(sample_num, 2).long()
    return values.to(device), X.to(device), Y.to(device)


def generate_sample_distribution_for_data_8(n):
    A = np.random.uniform(-0.2, 0.2, size=(n, n))
    cov = np.dot(A.T, A)
    mean = np.random.uniform(0, 1, size=n)
    return mean, cov
    
def generate_data_8(sample_num, n_agents, m_items, alpha, dx, dy, device, mu=0.5, means=None, covs=None):
    values = torch.zeros((sample_num, n_agents, m_items))

    # Generate two multivariate normal distributions
    for i in range(m_items):
        if means is not None:
            mean1, cov1 = means[0], covs[0]
            mean2, cov2 = means[1], covs[1]
        else:
            raise KeyError

        # Sample all values at once
        samples_d1 = np.random.multivariate_normal(mean1, cov1, size=sample_num)
        samples_d2 = np.random.multivariate_normal(mean2, cov2, size=sample_num)

        # Sample indicator for which distribution each sample should come from
        selector = np.random.rand(sample_num) < mu  # shape: (sample_num,)
        values_np = np.where(selector[:, None], samples_d1, samples_d2)  # shape: (sample_num, n)

        # Reshape to (sample_num, n_agents, m_items)
        values[:, :, i] = torch.tensor(values_np.reshape(sample_num, n_agents), dtype=torch.float32)

    values[values < 0] = 0
    values[values > 10] = 10
    # Agent and item index tensors
    X = torch.arange(n_agents).repeat(sample_num).reshape(sample_num, n_agents).long()
    Y = torch.arange(m_items).repeat(sample_num).reshape(sample_num, m_items).long()

    return values.to(device), X.to(device), Y.to(device)

import torch

def generate_data_9(sample_num, n_agents, m_items, alpha, dx, dy, device, beta=0.5, T_min=0.5, T_max=1.0):
    """
    基于“潜在总价值共享模型”生成估值数据。
    该模型能在支撑集（support）上产生负相关性。

    Args:
        sample_num (int): 需要生成的拍卖样本数量。
        n_agents (int): 竞标者（agent）的数量。
        m_items (int): 拍卖品（item）的数量。
        alpha (float): 对称狄利克雷分布的alpha参数。
                       - alpha < 1.0 导致强负相关 (分配不均)
                       - alpha > 1.0 导致弱负相关 (分配均匀)
        dx, dy: 保留参数，与您提供的格式保持一致。
        device: 指定张量所在的设备 (e.g., 'cpu' or 'cuda')。
        T_min (float): 潜在总价值分布的下界。
        T_max (float): 潜在总价值分布的上界。
    """
    values = torch.zeros((sample_num, n_agents, m_items))
    concentration = torch.full((n_agents,), float(beta))
    dirichlet_dist = torch.distributions.dirichlet.Dirichlet(concentration)

    for i in range(m_items):
        total_values = T_min + (T_max - T_min) * torch.rand(sample_num)

        # 从狄利克雷分布中生成价值份额 w_j
        # 形状: (sample_num, n_agents)
        shares = dirichlet_dist.sample((sample_num,))

        # 计算最终估值: v_ij = w_ij * T_j
        item_values = shares * total_values.unsqueeze(1)
        values[:, :, i] = item_values

    X = torch.arange(n_agents).repeat(sample_num).reshape(sample_num, n_agents).long()
    Y = torch.arange(m_items).repeat(sample_num).reshape(sample_num, m_items).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_data_11(sample_num, n_agents, m_items, alpha, dx, dy, device): # only serves for n_agents=2
    if n_agents != 2 or m_items != 2:
        print("This valuation distribution only serves for 2 bidders!")
        raise KeyError
    values = torch.rand((sample_num, 2, 2))

    values[:, 1, 0] = 1 - values[:, 0, 1]
    values[:, 1, 1] = 1 - values[:, 0, 0]

    X = torch.arange(n_agents).repeat(sample_num).reshape(sample_num, n_agents).long()
    Y = torch.arange(m_items).repeat(sample_num).reshape(sample_num, m_items).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_data_20(sample_num, n_agents, m_items, alpha, dx, dy, device): # only serves for n_agents=2
    base_values = torch.rand((sample_num, 1, m_items))
    special_values = torch.rand((sample_num, n_agents, m_items))
    values = alpha * base_values + (1 - alpha) * special_values

    X = torch.arange(n_agents).repeat(sample_num).reshape(sample_num, n_agents).long()
    Y = torch.arange(m_items).repeat(sample_num).reshape(sample_num, m_items).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_data_21(sample_num, n_agents, m_items, alpha, dx, dy, device): # only serves for n_agents=2
    if n_agents != 2:
        print("This valuation distribution only serves for 2 bidders!")
        raise KeyError
    base_values = torch.rand((sample_num, 2, m_items))
    base_values[:, 1, :] = 1 - base_values[:, 0, :]

    special_values = torch.rand((sample_num, n_agents, m_items))
    values = alpha * base_values + (1 - alpha) * special_values

    # values[:, 1, :] = 1 - values[:, 0, :]
    X = torch.arange(n_agents).repeat(sample_num).reshape(sample_num, n_agents).long()
    Y = torch.arange(m_items).repeat(sample_num).reshape(sample_num, m_items).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_data_22(sample_num, n_agents, m_items, alpha, dx, dy, device): # only serves for n_agents=2
    if n_agents != 2:
        print("This valuation distribution only serves for 2 bidders!")
        raise KeyError
    base_values = torch.rand((sample_num, 2, m_items))
    base_values[:, 1, :] = 1 - base_values[:, 0, :]

    special_values = torch.rand((sample_num, n_agents, m_items))
    values = alpha * base_values + (1 - alpha) * special_values
    
    values[:, 1, :] = values[:, 1, :] / 4
    # values[:, 1, :] = 1 - values[:, 0, :]
    X = torch.arange(n_agents).repeat(sample_num).reshape(sample_num, n_agents).long()
    Y = torch.arange(m_items).repeat(sample_num).reshape(sample_num, m_items).long()
    return values.to(device), X.to(device), Y.to(device)

def generate_data_23(sample_num, n_agents, m_items, alpha, dx, dy, device):
    """
    Only for n_agents=2.
    Bidder 1 values ~ equal-revenue on [ε,1], via inverse-CDF sampling.
    Bidder 2 value = linearly decreasing from ε to 0 as Bidder 1 goes from ε to 1.
    """
    if n_agents != 2:
        raise ValueError("This valuation distribution only serves for 2 bidders!")
    # --- Bidder 1: equal-revenue distribution on [ε,1] ---
    # CDF F(v) ∝ (1 - ε/v) truncated to [ε,1] ⇒ inverse: v = ε / (1 - U*(1-ε))
    U = torch.rand(sample_num, m_items, device=device)
    v1 = alpha / (1 - U * (1 - alpha))
    # numerical safety: clamp any tiny overshoot due to float
    v1 = v1.clamp(max=1.0)

    # --- Bidder 2: negative linear to v1; maps v1=ε↦ε, v1=1↦0
    v2 = alpha * (1 - v1) / (1 - alpha)

    # stack into shape [sample_num, 2, m_items]
    values = torch.stack([v1, v2], dim=1)
    values = values * 100
    # agent and item indices (if you really need them)
    X = torch.arange(n_agents, device=device).unsqueeze(0).repeat(sample_num, 1)
    Y = torch.arange(m_items, device=device).unsqueeze(0).repeat(sample_num, 1)

    return values, X.long(), Y.long()


    
if __name__ == "__main__":
    means, covs = np.zeros(shape=(10, 2, 10)), np.zeros(shape=(10, 2, 10, 10))
    for i in range(10):
        means[i, 0], covs[i, 0] = generate_sample_distribution_for_data_8(10)
        means[i, 1], covs[i, 1] = generate_sample_distribution_for_data_8(10)

    np.save('./data/data_8_means.npy', means)
    np.save('./data/data_8_covs.npy', covs)

    # test_sample_num = 20000
    # means = np.load('./data/data_8_means.npy')
    # covs = np.load('./data/data_8_covs.npy')
    # for (bidder, item) in [(2, 2), (2, 5), (3, 5), (5, 3), (3, 2), (4, 2), (5, 2), (8, 2), (10, 2)]:
    #     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #             x = generate_data_8(test_sample_num, bidder, item, 1, 10, 10, 'cpu', means=means[seed, :item, :, :bidder], covs=covs[seed, :item, :, :bidder, :bidder])
    #             torch.save(x, f'./data/{bidder}_{item}_data_8_test_{seed}.pt')
