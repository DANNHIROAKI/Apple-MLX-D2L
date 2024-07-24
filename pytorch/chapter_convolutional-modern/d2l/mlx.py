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
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

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

# Copyright © 2023 Apple Inc.
import gzip
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from mlx import data as dx
from mlx.data.datasets.common import ensure_exists, urlretrieve_with_progress


def load_mnist(root=None, train=True):
    """Defined in :numref:`sec_fashion_mnist`"""
    """Load a buffer with the MNIST dataset.

    If the data doesn't exist download it and save it for the next time.

    Args:
        root (Path or str, optional): The directory to load/save the data. If
            none is given the ``~/.cache/mlx.data/mnist`` is used.
        train (bool): Load the training or test set.
    """
    def get_notebook_dir():
        return Path(os.getcwd()).parent / 'data' / 'FashionMNIST'

    if root is None:
        root = get_notebook_dir()
    else:
        root = Path(root)

    ensure_exists(root)

    def download():
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        filename = [
            [NamedTemporaryFile(), "training_images", "train-images-idx3-ubyte.gz"],
            [NamedTemporaryFile(), "test_images", "t10k-images-idx3-ubyte.gz"],
            [NamedTemporaryFile(), "training_labels", "train-labels-idx1-ubyte.gz"],
            [NamedTemporaryFile(), "test_labels", "t10k-labels-idx1-ubyte.gz"],
        ]

        mnist = {}
        for out_file, _, name in filename:
            urlretrieve_with_progress(base_url + name, out_file.name)

        for out_file, key, _ in filename[:2]:
            with gzip.open(out_file.name, "rb") as f:
                mnist[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                    -1, 28, 28, 1
                )
        for out_file, key, _ in filename[-2:]:
            with gzip.open(out_file.name, "rb") as f:
                mnist[key] = np.frombuffer(f.read(), np.uint8, offset=8)
        train_set = [
            {"image": mnist["training_images"][i], "label": mnist["training_labels"][i]}
            for i in range(len(mnist["training_images"]))
        ]
        test_set = [
            {"image": mnist["test_images"][i], "label": mnist["test_labels"][i]}
            for i in range(len(mnist["test_images"]))
        ]

        with (root / "train.pkl").open("wb") as f:
            pickle.dump(train_set, f)
        with (root / "test.pkl").open("wb") as f:
            pickle.dump(test_set, f)

    if not (root / "test.pkl").is_file():
        download()

    pkl_file = (root / "train.pkl") if train else (root / "test.pkl")
    with pkl_file.open("rb") as f:
        return dx.buffer_from_vector(pickle.load(f))

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

def get_dataloader_workers():
    """使用4个进程来读取数据

    Defined in :numref:`sec_fashion_mnist`"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """Defined in :numref:`sec_fashion_mnist`"""
    """下载Fashion-MNIST数据集，然后将其加载到内存中

    Defined in :numref:`sec_fashion_mnist`"""
    mnist_train = load_mnist(root=None, train=True)
    mnist_test = load_mnist(root=None, train=False)

    train_iter = (
        mnist_train
        .to_stream()
        .batch(batch_size)
        .key_transform("image", lambda x: x.astype("float32") / 255) # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
        .prefetch(1, 4) # use 4 threads
    )

    test_iter = (
        mnist_test
        .to_stream()
        .batch(batch_size)
        .key_transform("image", lambda x: x.astype("float32") / 255) # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
        .prefetch(1, 4) # use 4 threads
    )
    if resize:
        train_iter = (
            mnist_train
            .to_stream()
            .batch(batch_size)
            .image_resize("image", resize, resize) # image must be of type UInt8
            .key_transform("image", lambda x: x.astype("float32") / 255) # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
            .prefetch(1, 4) # use 4 threads
        )
        train_iter = (
            mnist_test
            .to_stream()
            .batch(batch_size)
            .image_resize("image", resize, resize) # image must be of type UInt8
            .key_transform("image", lambda x: x.astype("float32") / 255) # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
            .prefetch(1, 4) # use 4 threads
        )

    return train_iter, test_iter

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
    for batch in data_iter:
        X, y = mx.array(batch["image"]).reshape((-1, 28 * 28)), mx.array(batch["label"]) # reshape image to match the W's shape
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
    if isinstance(train_iter, mlx.data._c.Stream):
        for batch in train_iter:
            X, y = mx.array(batch["image"]).reshape((-1, 28 * 28)), mx.array(batch["label"]) # reshape image to match the W's shape
            # 计算梯度并更新参数
            if isinstance(updater, optim.Optimizer):
                # 使用PyTorch内置的优化器和损失函数
                def loss_fn_mean(net, X, y):
                    y_hat = net(X)
                    return loss(y_hat, y, reduction="none").mean()
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
    else:
        for X, y in train_iter:
            def loss_fn(net, X, y):
                y_hat = net(X)
                return loss(y_hat, y, reduction="none").sum()
            loss_and_grad_fn = nn.value_and_grad(net, loss_fn)
            l, grads = loss_and_grad_fn(net, X, y)
            updater.update(net, grads)
            mx.eval(net.parameters())
            y_hat = net(X)
            l_sum = loss(y_hat, y, reduction="none").sum()
            metric.add(float(l_sum.item()), d2l.accuracy(y_hat, y), y.size)
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
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
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
    for batch in test_iter:
        X, y = mx.array(batch["image"]), mx.array(batch["label"])
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

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中

    Defined in :numref:`sec_text_preprocessing`"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元

    Defined in :numref:`sec_text_preprocessing`"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率

    Defined in :numref:`sec_text_preprocessing`"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表

    Defined in :numref:`sec_text_preprocessing`"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列

    Defined in :numref:`sec_language_model`"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    start_idx = np.random.randint(0, num_steps - 1)
    corpus = corpus[start_idx:]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield mx.array(X), mx.array(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列

    Defined in :numref:`sec_language_model`"""
    # 从随机偏移量开始划分序列
    offset = np.random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = mx.array(corpus[offset: offset + num_tokens])
    Ys = mx.array(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """Defined in :numref:`sec_language_model`"""
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表

    Defined in :numref:`sec_language_model`"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


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

