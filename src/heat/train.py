import os
import torch.distributed as dist

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from .equation import HeatEquation

from .derivative_wrapper import build_wrapper

from ..model import MLP
from .data import HeatDataset
from src.utils.glob import setup_logging, config
from src.utils import build_lr
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np


def train(rank, world_size, config):
    cfg = config.heat
    setup(rank, world_size)
    logger = setup_logging()
    logger.info(f"starting {rank}")

    layers = [cfg.equation.x_dim+1] + [cfg.model.width]*(cfg.model.depth-1) + [1]
    g = MLP(layers).to(rank)

    if world_size == 1:
        ddp_g = g
    else:
        ddp_g = DDP(g, device_ids=[rank])

    f = build_wrapper(cfg, ddp_g)

    dataset = HeatDataset(domain_bsz=cfg.train.batch.domain_size, \
        init_bsz=cfg.train.batch.initial_size, \
        spatial_bound_bsz=cfg.train.batch.spatial_boundary_size, \
        xdim=cfg.equation.x_dim, T=cfg.equation.T, rank=rank \
        )
    heat = HeatEquation(cfg.equation.x_dim)

    # test set
    if rank == 0:
        test_X, test_Y = generate_test_set(cfg, heat, rank)
        loss_list = []

    optimizer, scheduler = build_lr(ddp_g, cfg.train, cfg.train.iteration)

    def train_with_lbfgs(f, ddp_g, cfg, dataset, heat: HeatEquation, initial_lr=1.0, max_iter=50):
        optimizer = torch.optim.LBFGS(ddp_g.parameters(), lr=initial_lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            domain_X, init_X, spatial_boundary_X = dataset.get_online_data()
        
            dloss = heat.domain_loss(domain_X, f)
            iloss = heat.initial_loss(init_X, f)
            sloss = heat.spatial_boundary_loss(spatial_boundary_X, f)
            loss = cfg.train.loss.domain*dloss + cfg.train.loss.initial*iloss + cfg.train.loss.spatial_boundary*sloss
            loss_list.append(loss.detach().cpu().item())
            loss.backward()
            print(loss.detach().cpu().item())
            return loss

        optimizer.step(closure)

    for i in range(cfg.train.iteration):
        f.train()
        optimizer.zero_grad()

        domain_X, init_X, spatial_boundary_X = dataset.get_online_data()
        
        dloss = heat.domain_loss(domain_X, f)
        iloss = heat.initial_loss(init_X, f)
        sloss = heat.spatial_boundary_loss(spatial_boundary_X, f)
        loss = cfg.train.loss.domain*dloss + cfg.train.loss.initial*iloss + cfg.train.loss.spatial_boundary*sloss
        loss_list.append(loss.detach().cpu().item())
        if rank == 0:
            '''
            logger.info(f'iteration {i}\t| loss {loss.detach().cpu().item():.5f}\t| '
                f'domain {dloss.detach().cpu().item():.5f}\t|'
                f'initial {iloss.detach().cpu().item():.5f}\t|'
                f'spatial boundary {sloss.detach().cpu().item():.5f}\t'
            )'''

        if cfg.model.derivative != 'gt':
            loss.backward()
            optimizer.step()
            scheduler.step()

        if rank==0 and (i+1)%cfg.train.iteration==0:
            #LBFGS
            #print("Switching to L-BFGS optimizer.")
            #train_with_lbfgs(f, ddp_g, cfg, dataset, heat, initial_lr=1.0, max_iter=50)

            print("loss list: ", loss_list)
            
            x = np.linspace(0, 10.0, 50)
            t = np.linspace(0, 1.0, 10)
            xx, tt = np.meshgrid(x, t)
            X_test_np = np.vstack((xx.flatten(), tt.flatten())).T
            X_test_tor = torch.tensor(X_test_np, dtype=torch.float32).to(rank)
            u_true = heat.ground_truth(X_test_tor)
            u_true = u_true.cpu().detach().numpy().reshape(xx.shape)
            with torch.no_grad():
                f.eval()
            u_pred = f(X_test_tor)
            u_pred = u_pred.cpu().detach().numpy().reshape(xx.shape)
            u_true_str = '[\n' + ',\n'.join(['    [' + ', '.join(f"{x:.7f}" for x in row) + ']' for row in u_true]) + '\n]'
            u_pred_str = '[\n' + ',\n'.join(['    [' + ', '.join(f"{x:.7f}" for x in row) + ']' for row in u_pred]) + '\n]'
            print(f"u_true: {u_true_str}")
            print(f"u_pred: {u_pred_str}")
            
            with torch.no_grad():
                f.eval()

            x1 = np.linspace(0, 10.0, 50)
            t1 = np.ones(50) * 0.25
            xt1 = np.column_stack((x1, t1))
            xt1_tor = torch.tensor(xt1, dtype=torch.float32).to(rank)
            y_pred1 = f(xt1_tor)
            y_pred1 = y_pred1.cpu().detach().numpy().squeeze()
            y_true1 = heat.ground_truth(xt1_tor)
            y_true1 = y_true1.cpu().detach().numpy().squeeze()

            x2 = np.linspace(0, 10, 50)
            t2 = np.ones(50) * 0.5
            xt2 = np.column_stack((x2, t2))
            xt2_tor = torch.tensor(xt2, dtype=torch.float32).to(rank)
            y_pred2 = f(xt2_tor)
            y_pred2 = y_pred2.cpu().detach().numpy().squeeze()
            y_true2 = heat.ground_truth(xt2_tor)
            y_true2 = y_true2.cpu().detach().numpy().squeeze()

            x3 = np.linspace(0, 10, 50)
            t3 = np.ones(50) * 0.75
            xt3 = np.column_stack((x3, t3))
            xt3_tor = torch.tensor(xt3, dtype=torch.float32).to(rank)
            y_pred3 = f(xt3_tor)
            y_pred3 = y_pred3.cpu().detach().numpy().squeeze()
            y_true3 = heat.ground_truth(xt3_tor)
            y_true3 = y_true3.cpu().detach().numpy().squeeze()
            
            y_pred1_str = ', '.join([f"{x:.7f}" for x in y_pred1])
            y_pred2_str = ', '.join([f"{x:.7f}" for x in y_pred2])
            y_pred3_str = ', '.join([f"{x:.7f}" for x in y_pred3])
            y_true1_str = ', '.join([f"{x:.7f}" for x in y_true1])
            y_true2_str = ', '.join([f"{x:.7f}" for x in y_true2])
            y_true3_str = ', '.join([f"{x:.7f}" for x in y_true3])
            print(f"y_pred1: [{y_pred1_str}]")
            print(f"y_pred2: [{y_pred2_str}]")
            print(f"y_pred3: [{y_pred3_str}]")
            print(f"y_true1: [{y_true1_str}]")
            print(f"y_true2: [{y_true2_str}]")
            print(f"y_true3: [{y_true3_str}]")

            obser_t = np.linspace(0,1.0,100)
            observe_x = np.array([])
            observe_t = np.array([])
            for i in range(100):
                observe_x = np.concatenate((observe_x, np.linspace(0.0,10.0,100)))
                observe_t = np.concatenate((observe_t, np.full((100),obser_t[i])))

            observe_xt = np.vstack((observe_x, observe_t)).T
            obser_tor = torch.tensor(observe_xt, dtype=torch.float32)
            observe_y = f(obser_tor)
            observe_y = observe_y.cpu().detach().numpy().squeeze()
            obser_y_str = ', '.join([f"{x:.7f}" for x in observe_y])
            print(f"observe_y: [{obser_y_str}]")
            '''
            i_avg_err, i_rel_err = test(cfg, f, test_X, test_Y, rank, norm_type='l1')
            logger.info(f'L1 test error: average {i_avg_err}, relative {i_rel_err}')
            i_avg_err, i_rel_err = test(cfg, f, test_X, test_Y, rank, norm_type='l2')
            logger.info(f'L2 test error: average {i_avg_err}, relative {i_rel_err}')'''


    cleanup(rank, world_size)

def pgd(x, f, loss_func, step_cnt=5, step_size=0.2, t_lower_bound=0.0, t_upper_bound=1.0):
    for _ in range(step_cnt):
        x.requires_grad_()
        loss = loss_func(x, f)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size * torch.sign(grad.detach())
        x[:,-1] = torch.clamp(x[:,-1], t_lower_bound, t_upper_bound)
    return x

def generate_test_set(cfg, heat: HeatEquation, rank):
    x = 10 * torch.rand((cfg.test.total_size, heat.xdim), device=rank)
    test_X = torch.concat(
        [x, 
         torch.rand((cfg.test.total_size, 1), device=rank)*cfg.equation.T, # t ~ U(0,T)
        ],
        dim=1
    )
    test_Y = heat.ground_truth(test_X)
    return test_X, test_Y

def test(cfg, f, X, Y, rank, norm_type='l1'):
    if norm_type == 'l1':
        return test_l1(cfg, f, X, Y, rank)
    elif norm_type == 'l2':
        return test_l2(cfg, f, X, Y, rank)
    else:
        raise NotImplementedError
    
def test_l1(cfg, f, X, Y, rank):
    with torch.no_grad():
        f.eval()
        dataloader = DataLoader(TensorDataset(X, Y), batch_size=cfg.test.batch_size)
        tot_err, tot_norm = 0, 0
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            pred_y = f(x).squeeze()
            err = (pred_y - y).abs().sum()
            y_norm = y.abs().sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        avg_err = tot_err/X.shape[0]
        rel_err = tot_err/tot_norm
    return avg_err, rel_err

def test_l2(cfg, f, X, Y, rank):
    with torch.no_grad():
        f.eval()
        dataloader = DataLoader(TensorDataset(X, Y), batch_size=cfg.test.batch_size)
        tot_err, tot_norm = 0, 0
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            pred_y = f(x).squeeze()
            err = ((pred_y - y)**2).sum()
            y_norm = (y**2).sum()
            tot_err += err.cpu().item()
            tot_norm += y_norm
        tot_err, tot_norm = tot_err**0.5, tot_norm**0.5
        avg_err = tot_err/(X.shape[0]**0.5)
        rel_err = tot_err/tot_norm
    return avg_err, rel_err

def setup(rank, world_size):
    if world_size<=1:
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup(rank, world_size):
    if world_size<=1:
        return
    dist.destroy_process_group()

def heat_training():
    if config.heat.gpu_cnt == 1:
        train(0, 1, config)
    else:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= config.heat.gpu_cnt, \
            f"Requires at least {config.heat.gpu_cnt} GPUs to run, but got {n_gpus}"
        world_size = n_gpus

        mp.spawn(train,
                args=(world_size, config),
                nprocs=world_size,
                join=True)
