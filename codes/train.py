import os
import attrs
import torch
import argparse

# import numpy as np
from numpy import linspace
from torch.utils.data import DataLoader
from collections import defaultdict
# import mlflow
# from mlflow import MlflowClient
# from mlflow.entities import ViewType

from codes.utils import fix_seed
from codes.logger import MLFlowLogger

import codes.models as models
import codes.datasets as datasets
import codes.optimizers as optimizers


def metrics(model, loader, prefix, criterion, device, classes=False):
    correct, total_loss = 0, 0

    model.eval()
    for batch in loader:
        batch = [data.to(device) for data in batch]
        output = model(*batch[:-1])
        if hasattr(output, "logits"):
            output = output.logits
        loss = criterion(output, batch[-1])
        total_loss += loss.item()
        if classes:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(batch[-1].view_as(pred)).sum().item()

    m = defaultdict(float)
    m[prefix+'-loss'] = total_loss / len(loader)
    if classes:
        # print(f"{len(loader.dataset)=}")
        m[prefix+'-accuracy'] = 100. * correct / len(loader.dataset)
    return m


def train(cfg):
    config = attrs.asdict(cfg)
    logger = MLFlowLogger()
    if logger.check_exist(config):
        return
    logger.enabled = eval(os.environ['MLFLOW_VERBOSE'])
    logger.init(config)

    fix_seed(cfg.seed)
    device = torch.device(os.environ["TORCH_DEVICE"])

    cfg = argparse.Namespace(**config)

    train_data, shapes = getattr(datasets, cfg.dataset['name'])(cfg, train=True)

    dl_kwargs = {'batch_size': cfg.batchsize, 'shuffle': True,  # 'pin_memory': True,
                 'num_workers': 0}
    train_loader = DataLoader(train_data, **dl_kwargs)

    # shapes = (None, 7)
    classes = True

    model = getattr(models, cfg.model['name'])(*shapes).to(device)
    criterion = getattr(torch.nn, cfg.loss['name'])().to(device)

    if cfg.optimizer is not None:
        Optimizer = getattr(optimizers, cfg.optimizer['name'])
        optimizer = Optimizer(model.parameters(), cfg, device)
    else:
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.lr, eps=cfg.eps)

    test_data, val_data = getattr(datasets, cfg.dataset['name'])(cfg, train=False)
    dl_kwargs.update({'batch_size': 200})  # any number fitting memmory
    test_loader = DataLoader(test_data, **dl_kwargs)

    if logger.enabled:
        m = defaultdict(float)
        m.update(metrics(model, train_loader, 'train', criterion, device, classes))
        m.update(metrics(model, test_loader, 'test', criterion, device, classes))
        # m.update(optimizer.metrics())
        logger.log_metrics(m, 0)

    nticks = min(cfg.nepochs, 50)
    # log_ticks = np.linspace(0, cfg.nepochs, nticks, endpoint=True).round().astype(int)
    log_ticks = linspace(1, cfg.nepochs, nticks, endpoint=True).round().astype(int)
    steps = 0
    for e in range(1, cfg.nepochs+1):
        model.train()
        running_loss, correct, samples_per_epoch = 0, 0, 0
        train_iter = iter(train_loader)
        iter_steps = len(train_loader)
        for _ in range(iter_steps):
            batch = next(train_iter)
            batch = [data.to(device) for data in batch]

            output = model(*batch[:-1])
            if hasattr(output, "logits"):
                output = output.logits
            loss = criterion(output, batch[-1])

            running_loss += loss.item() * len(batch[-1])

            if logger.enabled and classes:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(batch[-1].view_as(pred)).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

        if logger.enabled and e in log_ticks:
            m.update(metrics(model, train_loader, 'train', criterion, device, classes))
            m.update(metrics(model, test_loader, 'test', criterion, device, classes))

            # m.update(optimizer.metrics())
            logger.log_metrics(m, steps)

    logger.terminate()