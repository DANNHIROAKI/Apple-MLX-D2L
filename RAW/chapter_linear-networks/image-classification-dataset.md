# 图像分类数据集
:label:`sec_fashion_mnist`

(**MNIST数据集**) :cite:`LeCun.Bottou.Bengio.ea.1998`
(**是图像分类中广泛使用的数据集之一，但作为基准数据集过于简单。
我们将使用类似但更复杂的Fashion-MNIST数据集**) :cite:`Xiao.Rasul.Vollgraf.2017`。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import sys
import paddle
from paddle.vision import transforms

d2l.use_svg_display()
```

```{.python .input}
#@tab mlx
%matplotlib inline
import random
import math
import mlx.core as mx
import numpy as np
import torch
import torchvision
from torchvision import transforms
from d2l import mlx as d2l

d2l.use_svg_display()
```

## 读取数据集

我们可以[**通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中**]。

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

```{.python .input}
#@tab paddle
trans = transforms.ToTensor()
mnist_train = paddle.vision.datasets.FashionMNIST(mode="train",
                                                  transform=trans)
mnist_test = paddle.vision.datasets.FashionMNIST(mode="test", transform=trans)
```

```{.python .input}
# @tab mlx
class Dataset: #@save
    def __init__(self, *tensors):
        assert all(
            tensors[0].shape[0] == tensor.shape[0] for tensor in tensors
        ), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(mx.array(tensor[index]) for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

class DataLoader: #@save
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
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
```

```{.python .input}
# @tab mlx
class MLX_Reshape(torch.nn.Module): #@save
    def forward(self, x):
        return x.permute(1, 2, 0)
    
class FashionMNIST:  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
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
```

Fashion-MNIST由10个类别的图像组成，
每个类别由*训练数据集*（train dataset）中的6000张图像
和*测试数据集*（test dataset）中的1000张图像组成。
因此，训练集和测试集分别包含60000和10000张图像。
测试数据集不会用于训练，只用于评估模型性能。

```{.python .input}
#@tab mxnet, pytorch, paddle
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

```{.python .input}
#@tab mlx
data = FashionMNIST()
len(data.train), len(data.val)
```

每个输入图像的高度和宽度均为28像素。
数据集由灰度图像组成，其通道数为1。
为了简洁起见，本书将高度$h$像素、宽度$w$像素图像的形状记为$h \times w$或（$h$,$w$）。

```{.python .input}
#@tab mxnet, pytorch, tensorflow, paddle
mnist_train[0][0].shape
```

```{.python .input}
#@tab mlx
data.train[0][0].shape
```

[~~两个可视化数据集的函数~~]

Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
以下函数用于在数字标签索引及其文本名称之间进行转换。

```{.python .input}
#@tab mxnet, pytorch, tensorflow, paddle
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

```{.python .input}
#@tab mlx
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i.item())] for i in labels]
```

我们现在可以创建一个函数来可视化这些样本。

```{.python .input}
#@tab mxnet, tensorflow
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab pytorch
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab paddle
#@save
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if paddle.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab mlx
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
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
```

以下是训练数据集中前[**几个样本的图像及其相应的标签**]。

```{.python .input}
X, y = mnist_train[:18]

print(X.shape)
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab paddle
X, y = next(iter(paddle.io.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape([18, 28, 28]), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab mlx
X, y = next(iter(d2l.DataLoader(data.train, batch_size=18, shuffle=True)))
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y))
```

## 读取小批量

为了使我们在读取训练集和测试集时更容易，我们使用内置的数据迭代器，而不是从零开始创建。
回顾一下，在每次迭代中，数据加载器每次都会[**读取一小批量数据，大小为`batch_size`**]。
通过内置数据迭代器，我们可以随机打乱了所有样本，从而无偏见地读取小批量。

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """在非Windows的平台上，使用4个进程来读取数据"""
    return 0 if sys.platform.startswith('win') else 4

# 通过ToTensor实例将图像数据从uint8格式变换成32位浮点数格式，并除以255使得所有像素的数值
# 均在0～1之间
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

```{.python .input}
#@tab paddle
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = paddle.io.DataLoader(dataset=mnist_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  return_list=True,
                                  num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab mlx
batch_size = 256
train_iter = d2l.DataLoader(data.train, batch_size, shuffle=True)
```

我们看一下读取训练数据所需的时间。

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## 整合所有组件

现在我们[**定义`load_data_fashion_mnist`函数**]，用于获取和读取Fashion-MNIST数据集。
这个函数返回训练集和验证集的数据迭代器。
此外，这个函数还接受一个可选参数`resize`，用来将图像大小调整为另一种形状。

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # 将所有数字除以255，使所有像素值介于0和1之间，在最后添加一个批处理维度，
    # 并将标签转换为int32。
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

```{.python .input}
#@tab paddle
#@save
def load_data_fashion_mnist(batch_size, resize=None):  
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = paddle.vision.datasets.FashionMNIST(mode="train",
                                                      transform=trans)
    mnist_test = paddle.vision.datasets.FashionMNIST(mode="test",
                                                     transform=trans)
    return (paddle.io.DataLoader(dataset=mnist_train,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 return_list=True,
                                 num_workers=get_dataloader_workers()),
            paddle.io.DataLoader(dataset=mnist_test,
                                 batch_size=batch_size,
                                 return_list=True,
                                 shuffle=True,
                                 num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab mlx
def load_data_fashion_mnist(batch_size, resize=None): #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中

    Defined in :numref:`sec_fashion_mnist`"""
    data = FashionMNIST(batch_size, resize)

    return (d2l.DataLoader(data.train, batch_size, shuffle=True),
            d2l.DataLoader(data.val, batch_size, shuffle=False))
```

下面，我们通过指定`resize`参数来测试`load_data_fashion_mnist`函数的图像大小调整功能。

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

我们现在已经准备好使用Fashion-MNIST数据集，便于下面的章节调用来评估各种分类算法。

## 小结

* Fashion-MNIST是一个服装分类数据集，由10个类别的图像组成。我们将在后续章节中使用此数据集来评估各种分类算法。
* 我们将高度$h$像素，宽度$w$像素图像的形状记为$h \times w$或（$h$,$w$）。
* 数据迭代器是获得更高性能的关键组件。依靠实现良好的数据迭代器，利用高性能计算来避免减慢训练过程。

## 练习

1. 减少`batch_size`（如减少到1）是否会影响读取性能？
1. 数据迭代器的性能非常重要。当前的实现足够快吗？探索各种选择来改进它。
1. 查阅框架的在线API文档。还有哪些其他数据集可用？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1788)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1787)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1786)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11692)
:end_tab:
