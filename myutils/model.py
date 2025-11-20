"""
-*- coding:utf-8 -*-
@Time      :2025/11/1 下午4:07
@Author    :Chen Junpeng

"""
import logging
import os.path
import warnings
from typing import List
import math
import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .file_process import *
from torch import nn
from torch.nn import init
from .utils_waring import *
from .plots import *


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, dropout, norm, activation, num_hidden_layers: int = None, hidden_size: int | List[int] = None, hidden_size_delay_rate: float = 0.5, logger: logging.Logger = None):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.log = logger.info if logger else print

        if isinstance(hidden_size, list):
            warnings.warn(UtilsWarning('The hidden size list is given, parameter num_hidden_layers is ignored.'))

            layers = []

            for i, size in enumerate(hidden_size):
                if i == 0:
                    layers.append(nn.Linear(input_size, size))
                else:
                    layers.append(nn.Linear(hidden_size[i-1], size))

                self.initialize_weights(layers[-1], activation)
                self.add_norm_layer(norm, layers, size)
                self.add_activation(activation, layers)
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_size[-1], output_size))
            self.initialize_weights(layers[-1], None)

        elif isinstance(hidden_size, int):
            assert num_hidden_layers is not None, "num_hidden_layers should be specified when hidden_size is an integer"
            assert num_hidden_layers > 0, "num_hidden_layers should be greater than 0"
            assert hidden_size_delay_rate > 0, "hidden_size_delay_rate should be greater than 0"

            layers = []

            for i in range(num_hidden_layers):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                    _size = hidden_size
                else:
                    now_size = max(int(_size * hidden_size_delay_rate), output_size)
                    layers.append(nn.Linear(_size, now_size))
                    _size = now_size

                self.initialize_weights(layers[-1], activation)
                self.add_norm_layer(norm, layers, _size)
                self.add_activation(activation, layers)
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(_size, output_size))
            self.initialize_weights(layers[-1], None)

        elif hidden_size is None:
            layers = [nn.Linear(input_size, output_size)]
            self.initialize_weights(layers[-1], None)

        else:
            raise ValueError(f"Invalid hidden size: {hidden_size}, with type of {type(hidden_size)}")

        self.network = nn.Sequential(*layers)

        self.log(self.network)

    def forward(self, x):
        return self.network(x)

    def add_activation(self, activation, layers):
        activation_dict = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU()
        }
        if activation not in activation_dict.keys():
            raise ValueError(f"Invalid activation type: {activation}, choices are {activation_dict.keys()}")
        layers.append(activation_dict[activation])

    def initialize_weights(self, module, activation):
        if isinstance(module, nn.Linear):
            if activation == 'relu' or activation is None:
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                self.log(f'Initialize {module} with ReLU')
            elif activation == 'leakyrelu':
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                self.log(f'Initialize {module} with LeakyReLU')
            else:
                init.xavier_normal_(module.weight)
                self.log(f'Initialize {module} with {activation}')
            if module.bias is not None:
                init.constant_(module.bias, 0)
                self.log(f'Initialize {module} bias with 0')
        else:
            self.log(f'Ignore {module} initialization')

    def add_norm_layer(self, norm, layers, hidden_size):
        norm_dict = {
            'layernorm': nn.LayerNorm,
            'batchnorm': nn.BatchNorm1d,
            'instancenorm': nn.InstanceNorm1d
        }
        if norm not in norm_dict.keys():
            raise ValueError(f"Invalid norm type: {norm}, choices are {norm_dict.keys()}")
        layers.append(norm_dict[norm](hidden_size))


class MyDataset(Dataset):
    def __init__(self, x, y, task='regression'):
        assert len(x) == len(y), f"x and y should have same length, but got {len(x)}, {len(y)}"
        self.x = torch.as_tensor(x, dtype=torch.float32)
        if task =='regression':
            self.y = torch.as_tensor(y, dtype=torch.float32)
            if self.y.dim() == 1:
                self.y = self.y.unsqueeze(1)
        elif task == 'classification':
            self.y = torch.as_tensor(y, dtype=torch.long)
            if self.y.dim() > 1:
                self.y = self.y.view(-1)
        else:
            raise ValueError(f"Invalid task type: {task}, choices are ['regression', 'classification']")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EarlyStop:
    def __init__(self, patience: int = 50, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if not math.isfinite(val_loss):
            print(f"[Warning] Metric is {val_loss}, ignoring...")
            return False, False

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epochs_no_improve = 0
        return self.early_stop, self.epochs_no_improve == 0


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        mse = ((x - y) ** 2).mean()
        rmse = torch.sqrt(mse)
        return rmse


class MyTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 optimizer: str,
                 scheduler: str,
                 lr: float,
                 loss_fn: str,
                 epochs: int,
                 batch_size: int,
                 device: str,
                 save_path: str,
                 early_stop_patience: int = 50,
                 early_stop_delta: float = 0,
                 save_all_ckpts: bool = False,
                 test_dataset: Dataset = None,
                 optimizer_kwargs: dict = None,
                 scheduler_kwargs: dict = None,
                 logger: logging.Logger = None,
                 **kwargs):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.loss_fn_name = loss_fn
        self.scheduler_name = scheduler
        self.optimizer = self.get_optimizer(optimizer, **self.optimizer_kwargs)
        self.scheduler = self.get_scheduler(scheduler, **self.scheduler_kwargs)
        self.loss_fn = self.get_loss_fn()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset if test_dataset is not None else None
        self.batch_size = batch_size
        self.log = logger.info if logger else print
        self.kwargs = kwargs
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size) if test_dataset is not None else None
        self.early_stop = EarlyStop(early_stop_patience, early_stop_delta)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.save_all_ckpt = save_all_ckpts

    def get_optimizer(self, optimizer_name, **kwargs):
        OPTIMIZER_DICT = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD
        }
        if optimizer_name not in OPTIMIZER_DICT.keys():
            raise ValueError(f"Invalid optimizer type: {optimizer_name}, choices are {OPTIMIZER_DICT.keys()}")
        return OPTIMIZER_DICT[optimizer_name](self.model.parameters(), lr=self.lr, **kwargs)

    def get_scheduler(self, scheduler_name, **kwargs):
        """

        :param scheduler_name:
        :param kwargs:
            steplr: step_size, gamma
            cosineannealinglr: T_max, eta_min
            cosineannealingwarmrestarts: T_0, T_mult, eta_min
            reducelronplateau: mode, factor, patience
        :return:
        """
        SCHEDULER_DICT = {
            'steplr': lr_scheduler.StepLR,
            'cosineannealinglr': lr_scheduler.CosineAnnealingLR,
            'cosineannealingwarmrestarts': lr_scheduler.CosineAnnealingWarmRestarts,
            'reducelronplateau': lr_scheduler.ReduceLROnPlateau,
            'onecyclelr': lr_scheduler.OneCycleLR,
        }

        if scheduler_name == 'onecyclelr':
            kwargs.setdefault('steps_per_epoch', len(self.train_loader))
            kwargs.setdefault('epochs', self.epochs)
            kwargs.setdefault('max_lr', self.lr * 10)
        elif scheduler_name == 'cosineannealinglr':
            kwargs.setdefault('T_max', self.epochs)

        if scheduler_name not in SCHEDULER_DICT.keys():
            raise ValueError(f"Invalid scheduler type: {scheduler_name}, choices are {SCHEDULER_DICT.keys()}")
        return SCHEDULER_DICT[scheduler_name](self.optimizer, **kwargs)

    def get_loss_fn(self):
        LOSS_FN_DICT = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'rmse': RMSELoss,
            'huber': nn.HuberLoss,
            'bce': nn.BCEWithLogitsLoss,
            'crossentropy': nn.CrossEntropyLoss,
        }
        if self.loss_fn_name not in LOSS_FN_DICT.keys():
            raise ValueError(f"Invalid loss function type: {self.loss_fn_name}, choices are {LOSS_FN_DICT.keys()}")
        return LOSS_FN_DICT[self.loss_fn_name]()

    def train(self):
        for epoch in tqdm(range(self.epochs), desc='Training', total=self.epochs):
            self.model.train()
            train_loss = 0.0
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x.size(0)
                if self.scheduler_name == 'onecyclelr':
                    self.scheduler.step()

            train_loss /= len(self.train_dataset)
            self.log(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')

            self.model.eval()

            val_loss = 0.0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = self.model(x)
                    loss = self.loss_fn(y_pred, y)
                    val_loss += loss.item() * x.size(0)

                val_loss /= len(self.val_dataset)
                stop, save = self.early_stop(val_loss=val_loss)
                if stop:
                    self.log(f'Early stopping at epoch {epoch+1}')
                    break
                if self.scheduler_name == 'reducelronplateau':
                    self.scheduler.step(val_loss)
                elif self.scheduler_name != 'onecyclelr':
                    self.scheduler.step()

                if self.save_all_ckpt:
                    self.save_checkpoint(epoch=epoch+1)
                if save:
                    self.save_checkpoint()

                self.log(f'Epoch: {epoch+1}, Val Loss: {val_loss:.4f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}')

    def test(self):
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided")
        self.model.eval()
        y_pred_all = []
        y_true_all = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                y_pred_all.append(y_pred.cpu())
                y_true_all.append(y.cpu())

        y_pred_all = torch.cat(y_pred_all).numpy().flatten().tolist()
        y_true_all = torch.cat(y_true_all).numpy().flatten().tolist()

        save_array_to_npy_file(y_pred_all, self.save_path, 'y_pred')
        save_array_to_npy_file(y_true_all, self.save_path, 'y_true')
        set_plot_config(20, 28, 24, 20, 20)
        draw_parity_plot(y_pred_all, y_true_all, self.save_path, 'test', 'Test', msg=['mae', 'r2'], save_result=True)

    def save_checkpoint(self, epoch=None):
        if self.save_all_ckpt:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, f"model.ckpt.{epoch}"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, f"best_model.ckpt"))


def normalize(x, save_path, mode='minmax', params_file=None, logger=None):
    log = logger.info if logger else print
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"making dir {save_path}")

    x = np.array(x)

    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN or infinity")

    EPS = 1e-8

    if mode =='minmax':
        log(f"normalize data with minmax")
        if params_file is None:
            x_min = np.min(x, axis=0)
            x_max = np.max(x, axis=0)

            params = {'min': x_min, 'max': x_max}
            final_params_path = save_dict_to_npz_file(params, save_path, 'minmax_params', logger=logger)
        else:
            params = load_npz_file(params_file, logger=logger)
            if params is None:
                raise RuntimeError(f"Failed to load params from {params_file}")
            x_min = params['min']
            x_max = params['max']
            final_params_path = params_file

        x_norm = (x - x_min) / (x_max - x_min + EPS)

    elif mode == 'norm':
        log(f"normalize data with norm")
        if params_file is None:
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0)

            params = {'mean': x_mean, 'std': x_std}
            final_params_path = save_dict_to_npz_file(params, save_path, 'norm_params', logger=logger)
        else:
            params = load_npz_file(params_file, logger=logger)
            if params is None:
                raise RuntimeError(f"Failed to load params from {params_file}")
            x_mean = params['mean']
            x_std = params['std']
            final_params_path = params_file

        x_norm = (x - x_mean) / (x_std + EPS)

    else:
        raise ValueError(f"Invalid mode: {mode}, choices are ['minmax', 'norm']")

    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return x_norm, final_params_path


def denormalize(x_norm, params_file, mode='minmax', logger=None):
    log = logger.info if logger else print
    x_norm = np.array(x_norm)

    EPS = 1e-8

    if params_file is None:
        raise ValueError("params_file is required for denormalization.")

    params = load_npz_file(params_file, logger=logger)
    if params is None:
        raise RuntimeError(f"Failed to load params from {params_file}")

    if mode == 'minmax':
        if 'min' not in params or 'max' not in params:
            raise KeyError(f"Params file {params_file} does not contain 'min' or 'max' keys.")

        x_min = params['min']
        x_max = params['max']

        x_original = x_norm * (x_max - x_min + EPS) + x_min

    elif mode == 'norm':
        if 'mean' not in params or 'std' not in params:
            raise KeyError(f"Params file {params_file} does not contain 'mean' or 'std' keys.")

        x_mean = params['mean']
        x_std = params['std']

        x_original = x_norm * (x_std + EPS) + x_mean

    else:
        raise ValueError(f"Invalid mode: {mode}, choices are ['minmax', 'norm']")

    return x_original