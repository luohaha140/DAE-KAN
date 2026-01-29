import torch
import torch.nn as nn
import numpy as np
from model.KANLayer import KANLayer

class model_dyn(nn.Module):
    def __init__(self, width=None,device='cpu',grid=3,k=3,noise=1,seed=0):
        super(model_dyn, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.layer_size = width
        self.device = device
        self.grid = grid
        self.k = k
        self.noise = noise
        self.nn_dyn = nn.ModuleList()
        self.build_dyn()

    def build_dyn(self):
        self.U = KANLayer(in_dim=self.layer_size[0], out_dim=self.layer_size[1], num=self.grid, k=self.k,
                          noise_scale=self.noise, device=self.device)
        self.V = KANLayer(in_dim=self.layer_size[0], out_dim=self.layer_size[1], num=self.grid, k=self.k,
                          noise_scale=self.noise, device=self.device)
        for k in range(len(self.layer_size) - 2):
            self.nn_dyn.append(
                KANLayer(in_dim=self.layer_size[k], out_dim=self.layer_size[k + 1], num=self.grid, k=self.k,
                         noise_scale=self.noise, device=self.device)
            )
        self.nn_dyn.append(KANLayer(in_dim=self.layer_size[-2], out_dim=self.layer_size[-1], num=self.grid, k=self.k,
                                 noise_scale=self.noise, device=self.device))
    # def forward(self, input):
    #     y = input
    #     for i in range(len(self.nn_dyn)):
    #         y = self.nn_dyn[i](y)[0]
    #     return y
    def forward(self, input):
        """
        FNN forward pass
        Args:
            :input (Tensor): \in [B, d_in]
        Returns:
            :y (Tensor): \in [B, d_out]
        """
        y = input
        u = self.U(y)[0]
        v = self.V(y)[0]
        for i in range(len(self.nn_dyn) - 1):
            y = self.nn_dyn[i](y)[0]
            y = (1 - y) * u + y * v
        y = self.nn_dyn[-1](y)[0]
        return y
class model_alg(nn.Module):
    def __init__(self, width=None,device='cpu',grid=3,k=3,noise=1,scale_base_mu=0.0, scale_base_sigma=1.0,seed=0):
        super(model_alg, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.layer_size = width
        self.device = device
        self.grid = grid
        self.k = k
        self.noise = noise
        self.scale_base_mu = scale_base_mu
        self.scale_base_sigma = scale_base_sigma
        self.nn_dyn = nn.ModuleList()
        self.build_alg()
        # print(self.net)
    def build_alg(self):
        self.U = KANLayer(in_dim=self.layer_size[0], out_dim=self.layer_size[1], num=self.grid, k=self.k,
                          noise_scale=self.noise, device=self.device)
        self.V = KANLayer(in_dim=self.layer_size[0], out_dim=self.layer_size[1], num=self.grid, k=self.k,
                          noise_scale=self.noise, device=self.device)
        for k in range(len(self.layer_size) - 2):
            self.nn_dyn.append(
                KANLayer(in_dim=self.layer_size[k], out_dim=self.layer_size[k + 1], num=self.grid, k=self.k,
                         noise_scale=self.noise, device=self.device)
            )
        self.nn_dyn.append(KANLayer(in_dim=self.layer_size[-2], out_dim=self.layer_size[-1], num=self.grid, k=self.k,
                                    noise_scale=self.noise, device=self.device))
    # def forward(self, input):
    #     y = input
    #     for i in range(len(self.nn_dyn)):
    #         y = self.nn_dyn[i](y)[0]
    #     return y
    def forward(self, input):
        """
        FNN forward pass
        Args:
            :input (Tensor): \in [B, d_in]
        Returns:
            :y (Tensor): \in [B, d_out]
        """
        y = input
        u = self.U(y)[0]
        v = self.V(y)[0]
        for i in range(len(self.nn_dyn) - 1):
            y = self.nn_dyn[i](y)[0]
            y = (1 - y) * u + y * v
        y = self.nn_dyn[-1](y)[0]
        return y
class model_KAN(nn.Module):
    def __init__(
        self,
        width_dyn = None,grid_dyn = 3, k_dyn = 3, noise_dyn = 1,
        width_alg=None, grid_alg=3, k_alg=3, noise_alg=1
        ):
        super(model_KAN, self).__init__()
        self.Y = model_dyn(width=width_dyn,grid=grid_dyn,k=k_dyn,noise=noise_dyn)
        self.Z = model_alg(width=width_alg,grid=grid_alg,k=k_alg,noise=noise_alg)

    def forward(self, input):
        Y = self.Y(input)
        # Y0 = Y[:,0].unsqueeze(1)
        # Y1 = Y[:,1].unsqueeze(1)
        # Y2 = Y[:,2].unsqueeze(1)
        # Y3 = Y[:,3].unsqueeze(1)
        Z = self.Z(input)
        x = torch.cat((Y,Z),dim=1)
        return x