DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import mlx
import mlx.core
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

nn_Module = nn.Module

#################   WARNING   ################
# The below part is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

d2l = sys.modules[__name__]

import mlx.core as mx
import mlx.data
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声

    Defined in :numref:`sec_linear_scratch`"""
    X = mx.random.normal(shape=(num_examples, len(w)), loc=0.0, scale=1.0)
    y = mx.matmul(X, w) + b
    y += mx.random.normal(shape=y.shape, loc=0.0, scale=0.01)
    return X, y.reshape((-1, 1))

def linreg(X, params):
    """线性回归模型

    Defined in :numref:`sec_linear_scratch`"""
    return mx.matmul(X, params[0]) + params[1]

def squared_loss(y_hat, y):
    """均方损失

    Defined in :numref:`sec_linear_scratch`"""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2

def sgd(params, grads, lr, batch_size):
    """小批量随机梯度下降

    Defined in :numref:`sec_linear_scratch`"""
    for i in range(len(params)):
        params[i] -= lr * grads[i] / batch_size

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个mlx数据迭代器

    Defined in :numref:`sec_linear_concise`"""
    X = data_arrays[0]
    y = data_arrays[1]
    perm = mx.array(np.random.permutation(data_arrays[1].size))
    for s in range(0, y.size, batch_size):
        ids = perm[s:s + batch_size]
        yield X[ids], y[ids]

class Dataset:
    def __init__(self, *tensors):
        """Defined in :numref:`sec_fashion_mnist`"""
        assert all(
            tensors[0].shape[0] == tensor.shape[0] for tensor in tensors
        ), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(mx.array(tensor[index]) for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """Defined in :numref:`sec_fashion_mnist`"""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        end_index = self.current_index + self.batch_size
        if end_index > len(self.indices):
            if self.drop_last:
                raise StopIteration
            else:
                end_index = len(self.indices)

        batch_indices = self.indices[self.current_index:end_index]
        batch = [self.dataset[i] for i in batch_indices]
        self.current_index = end_index

        return self.collate_fn(batch)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

    def collate_fn(self, batch):
        if isinstance(batch[0], tuple):
            if (len(batch[0])) == 2:
                data, targets = zip(*batch)
                data = mx.array(data)
                targets = mx.array(targets)
                return data, targets
            if (len(batch[0])) == 4:
                data, decoder_input, src_valid_len, targets = zip(*batch)
                data = mx.array(data)
                decoder_input = mx.array(decoder_input)
                src_valid_len = mx.array(src_valid_len)
                targets = mx.array(targets)
                return data, decoder_input, src_valid_len, targets
        return mx.array(batch)

class MLX_Reshape(torch.nn.Module):
    """Defined in :numref:`sec_fashion_mnist`"""
    def forward(self, x):
        return x.permute(1, 2, 0)

class FashionMNIST:
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        """Defined in :numref:`sec_fashion_mnist`"""
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor(),
                                    MLX_Reshape()])
        self.train = torchvision.datasets.FashionMNIST(
            root="../data", train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root="../data", train=False, transform=trans, download=True)

        train_data = np.array([np.array(self.train[i][0]) for i in range(len(self.train))])
        train_targets = self.train.targets.numpy()
        val_data = np.array([np.array(self.val[i][0]) for i in range(len(self.val))])
        val_targets = self.val.targets.numpy()

        self.train = d2l.Dataset(train_data, train_targets)
        self.val = d2l.Dataset(val_data, val_targets)

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签

    Defined in :numref:`sec_fashion_mnist`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i.item())] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if type(img) == mx.array:
            # 图片张量
            ax.imshow(img)
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def load_data_fashion_mnist(batch_size, resize=None):
    """Defined in :numref:`sec_fashion_mnist`"""
    """下载Fashion-MNIST数据集，然后将其加载到内存中

    Defined in :numref:`sec_fashion_mnist`"""
    data = FashionMNIST(batch_size, resize)
    print("here!")

    return (d2l.DataLoader(data.train, batch_size, shuffle=True),
            d2l.DataLoader(data.val, batch_size, shuffle=False))

def accuracy(y_hat, y):
    """计算预测正确的数量

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype) == y
    return float(cmp.astype(y.dtype).sum().item()) # use item because could not convert string to float: array(1, dtype=int32)

def evaluate_accuracy(net, data_iter, params):
    """Defined in :numref:`sec_softmax_scratch`"""
    """计算在指定数据集上模型的精度
    Defined in :numref:`sec_softmax_scratch`"""
    if isinstance(net, nn.Module):
        net.train(False)
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        if isinstance(net, nn.Module):
            metric.add(accuracy(net(X), y), y.size)
        else:
            metric.add(accuracy(net(X, params), y), y.size)
        break
    return metric[0] / metric[1]

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net, train_iter, loss, updater, batch_size, params):
    """Defined in :numref:`sec_softmax_scratch`"""
    """训练模型一个迭代周期（定义见第3章）
    Defined in :numref:`sec_softmax_scratch`"""
    # 将模型设置为训练模式
    if isinstance(net, nn.Module):
        net.train(True)
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
        # 计算梯度并更新参数
    for X, y in train_iter:
        if isinstance(updater, optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            def loss_fn_mean(net, X, y):
                y_hat = net(X)
                return loss(y_hat, y).mean()
            loss_and_grad_fn = nn.value_and_grad(net, loss_fn_mean)
            l, grad = loss_and_grad_fn(net, X, y)
            updater.update(net, grad)
            mx.eval(net.parameters())
            y_hat = net(X)
            l_sum = loss(y_hat, y).sum()
            metric.add(float(l_sum.item()), accuracy(y_hat, y), y.size)
        else:
            # 使用定制的优化器和损失函数
            def loss_fn_sum(params):
                y_hat = net(X, params)
                return loss(y_hat, y).sum()
            loss_and_grad_fn = mx.value_and_grad(loss_fn_sum)
            l, grad = loss_and_grad_fn(params)
            updater(grad, X.shape[0])
            mx.eval(params)
            y_hat = net(X, params)
            metric.add(float(l.item()), accuracy(y_hat, y), y.size)
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, batch_size, params):
    """Defined in :numref:`sec_softmax_scratch`"""
    """训练模型（定义见第3章）
    Defined in :numref:`sec_softmax_scratch`"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, batch_size, params)
        test_acc = evaluate_accuracy(net, test_iter, params)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, params, n=6):
    """Defined in :numref:`sec_softmax_scratch`"""
    """预测标签（定义见第3章）
    Defined in :numref:`sec_softmax_scratch`"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    if isinstance(net, nn.Module):
        preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    else:
        preds = d2l.get_fashion_mnist_labels(net(X, params).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

def evaluate_loss(net, data_iter, loss, params):
    """评估给定数据集上模型的损失

    Defined in :numref:`sec_model_selection`"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        if isinstance(net, nn.Module):
            out = net(X)
            y = d2l.reshape(y, out.shape) # ?
            l = loss(out, y)
            metric.add(l.sum().item(), l.size)
        else:
            out = net(X, params)
            y = d2l.reshape(y, out.shape) # ?
            l = loss(out, y)
            metric.add(l.sum().item(), l.size)
    return metric[0] / metric[1]

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名

    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件

    Defined in :numref:`sec_kaggle_house`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件

    Defined in :numref:`sec_kaggle_house`"""
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

def corr2d(X, K):
    """计算二维互相关运算

    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y

def evaluate_accuracy_gpu(net, data_iter):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.train(False)
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        metric.add(d2l.accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr):
    """用GPU训练模型(在第六章定义)

    Defined in :numref:`sec_lenet`"""
    # 初始化权重
    def init_weights(array):
        if array.ndim > 1:
            weight_fn = nn.init.glorot_uniform()
            array = weight_fn(array)
        return array
    for module in net.layers:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            module.update(mlx.utils.tree_map(lambda x: init_weights(x), module.parameters()))
    print(mx.default_device())
    optimizer = optim.SGD(learning_rate=lr)
    loss = nn.losses.cross_entropy
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    # num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        if isinstance(net, nn.Module):
            net.train(True)
        for i, (X, y) in enumerate(train_iter):
            # 使用PyTorch内置的优化器和损失函数
            def loss_fn(net, X, y):
                y_hat = net(X)
                return loss(y_hat, y, reduction="none").mean()
            loss_and_grad_fn = nn.value_and_grad(net, loss_fn)
            l, grad = loss_and_grad_fn(net, X, y)
            optimizer.update(net, grad)
            mx.eval(net.parameters())
            y_hat = net(X)
            metric.add(l.item() * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, test_acc))
            # test_acc = evaluate_accuracy_gpu(net, test_iter)
            # animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ')


# Alias defined in config.ini
nn_Module = nn.Module

ones = mx.ones
zeros = mx.zeros
arange = mx.arange
meshgrid = mx.meshgrid
sin = mx.sin
sinh = mx.sinh
cos = mx.cos
cosh = mx.cosh
tanh = mx.tanh
linspace = mx.linspace
exp = mx.exp
log = mx.log
random.normal = mx.random.normal
random.randint = mx.random.randint
matmul = mx.matmul
int32 = mx.int32
float32 = mx.float32
concat = mx.concatenate
stack = mx.stack
abs = mx.abs
eye = mx.eye
tensor = mx.array
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)

