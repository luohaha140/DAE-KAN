import torch
import torch.nn as nn
import numpy as np
from model.spline import *
from utils.utils import sparse_mask


class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        noise_scale: float
            spline scale at initialization
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base_mu: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
        scale_base_sigma: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
        scale_sp: float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            the id of activation functions that are locked
        device: str
            device
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False):
        ''''
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_base_sigma : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_sp : float
                the scale of the base function spline(x).
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            device : str
                device
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.
            
        Returns:
        --------
            self

        '''
        super(KANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num

        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        
        self.grid_eps = grid_eps
        
        self.to(device)
        
    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs

        '''
        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
            
        base = self.base_fun(x) # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k, device=self.device)
        
        postspline = y.clone().permute(0,2,1)
            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
            
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postspline

