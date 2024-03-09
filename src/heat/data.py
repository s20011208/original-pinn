import torch
#时间坐标都是[0,1]
#空间坐标是[0,10.0]
class HeatDataset():
    def __init__(self, domain_bsz, init_bsz, spatial_bound_bsz, xdim=1, T=1.0, rank=0):
        self.domain_bsz = domain_bsz
        self.init_bsz = init_bsz
        self.spatial_bound_bsz = spatial_bound_bsz
        self.xdim = xdim
        self.T = T
        self.rank = rank

    def get_online_data(self):

        x = 10 * torch.rand((self.domain_bsz + self.init_bsz, self.xdim), device=self.rank)

        domain_X = torch.concat(
            [x[:self.domain_bsz], 
             torch.rand((self.domain_bsz, 1), device=self.rank)*self.T, 
            ],
            dim=1
        )
        
        #init这里是t=T,不是t=0
        init_X = torch.concat(
            [x[-self.init_bsz:], 
             torch.ones((self.init_bsz, 1), device=self.rank), # t = 1
            ],
            dim=1
        )

        # 生成形状为(self.spatial_bound_bsz, self.xdim)的随机浮点数张量
        x = torch.rand((self.spatial_bound_bsz, self.xdim), device=self.rank)

        # 将随机浮点数四舍五入到最近的整数，然后确保结果是浮点数类型
        x = torch.round(x).to(torch.float32)
        x = 10 * x

        spatial_boundary_X = torch.concat(
            [x,  
             torch.rand((self.spatial_bound_bsz, 1), device=self.rank)*self.T, # t ~ U(0,T)
            ],
            dim=1
        )

        return domain_X, init_X, spatial_boundary_X