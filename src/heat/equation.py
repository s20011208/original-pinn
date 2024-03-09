import torch
from scipy.stats import norm
import numpy as np

class HeatEquation:
    def __init__(self, x_dim) -> None:
        self.xdim = x_dim

    def domain_loss(self, X, f, sample_cnt=None):
        # dt: (batch, 1)
        # dx: (batch, 1)
        # dx2: (batch, 1)
        dt = f.dt(X, sample_cnt)
        dx2 = f.dx2(X, sample_cnt)
        dx = f.dx(X, sample_cnt)
        f_val = f.ff(X, sample_cnt)

        residual = dt.squeeze(1) - 4.5 * X[:,0]**2 * torch.sum(dx2, dim=1) + 0.03 * X[:,0] * torch.sum(dx, dim=1) - 0.03 * f_val.squeeze(1)
        loss = torch.mean(residual**2)
        return loss

    def initial_loss(self, X, f, sample_cnt=None):
        # u(x, 1) = max(x-4, 0)
       
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        gt = self.ground_truth(X)
        
        # Alternatively
        # x = X[:, :-1]
        # gt = torch.sum(x**2, dim=1) / (2*x.shape[1])

        return torch.mean((y-gt)**2)

    def spatial_boundary_loss(self, X, f, sample_cnt=None):
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        gt = self.ground_truth(X)
        return torch.mean((y-gt)**2)

    def ground_truth(self, X):
        _, total_dim = X.shape
        sigma = 3.0
        r = 0.03
        # X.shape: (batch_size, 2)
        # x.shape: (batch_size, 1)
        # t.shape: (batch_size)
        # gt.shape: (batch_size)
        x, t = X[:,0], X[:,1]
        
        d1 = (torch.log(x/4) + (r + 0.5*sigma**2)*(1 - t)) / (sigma*torch.sqrt(1 - t))
        d2 = (torch.log(x/4) + (r - 0.5*sigma**2)*(1 - t)) / (sigma*torch.sqrt(1 - t))
        Nd1 = 0.5 * (1 + torch.erf(d1 / torch.sqrt(torch.tensor(2.0))))
        Nd2 = 0.5 * (1 + torch.erf(d2 / torch.sqrt(torch.tensor(2.0))))
        C = x * Nd1 - 4 * torch.exp(r*(t - 1)) * Nd2
        gt = C.squeeze()
        
        return gt
