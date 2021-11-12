# PyTorch笔记

## [B, C, H, W]

* flatten

```python
x = torch.flatten(x, start_dim=1)  # dim_0 : batch_size
x = nn.Flatten()(x)  # default first dim to flatten = 1
```

* totensor

```python
# torchvision.transforms.ToTensor()

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

```

* showimg

```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转置回[H,W,C]
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()  # 得到第一批

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
```

* torch2array

```python
x_torch = torch.from_numpy(x_array)
```

* array2torch

```python
x_array = x_torch.numpy()
```

## train

* get device

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = x.to(device)
model = model.to(device)
```

* model.train() v.s. model.eval()

```python
def train(dataloader, model, loss_fn, optimizer):
 model.train()
 ...

def test(dataloader, model, loss_fn):
 model.eval()
 ...
 with torch.no_grad():
  ...
```

model.train() 与 model.eval() 的区别在于 dropout/batch normalization, 利用上下文管理器with可以**暂时**关掉grad的属性，加速减少显存

* crossentropy

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

model.train()
for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**问题**：pred (#batch, #C)， y (#batch)，如何求 **loss** ？

**公式**：$
\small{l = -\sum_{k=1}^C y^{one\_ hot}_k\cdot \log(pred_k)
= - \log(pred_y)
}$

**注意**： 如果分类任务使用nn.CrossEntropyLoss()，最后一层全连接之后可以不用添加Softmax，已经包含在loss的计算中
>Note that this case is equivalent to the combination of  [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax)  and  [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)

* BatchNormalization
[Batch Normalization原理与实战](https://zhuanlan.zhihu.com/p/34879333)
[什么是批标准化 (Batch Normalization)](https://zhuanlan.zhihu.com/p/24810318)
[Batch Normalization详解](https://www.cnblogs.com/shine-lee/p/11989612.html)

1. BN层自带学习属性($\gamma$以及$\beta$)，而且会保留均值及方差（考虑 **model.eval()** 时，输入的batch_szie可能为1，所有BN层的参数都是固定的，在训练过程中迭代更新），所以网络中的每个BN层都需要单独定义；这不同于ReLU，可以复用

2. BN层的操作是以 **feature** 为单位的：卷积就是 feature map 的个数，即有 **c_out** 对 $\mu,\sigma,\gamma,\beta$；全连接层就是神经元的个数

3. BN提出是为了解决激活函数分布趋于饱和端的情况，所以推荐使用在激活函数之前；又由于BN包含了 **shift** 和 **scaling** 的作用，所以可以将BN之前层的bias设为False

* LayerNormalization
[详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246)
[深度学习中的Normalization模型](https://www.jiqizhixin.com/articles/2018-08-29-7)

Normalization都在一个框架下，都包含有训练学习的思想

## model

* print model

```python
print(net)
```

实验发现，net 架构只与 __init__ 中定义的self.layer有关，即使一个层在forward中使用多次，也只会打印一次

* print parameters

```python
# named_parameters
for name, param in net.named_parameters():
 print("name: {:<15}, shape: {}".format(name, param.shape))

# parameters
for param in net.parameters():
 print("shape: {}".format(param.shape))
```

查看源码发现，net.parameters() 就是调用的 net.named_parameters()，所以以后还是用后者吧

## dataset

参考 [**霹雳吧啦Wz**](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/data_set) 的方法，利用标准库 **os**，可以分割制作出自己的数据集

* 制作数据集

```python
import os
from shutil import copyfile, rmtree
import random


def mk_dir(dir):
    if os.path.exists(dir):
        rmtree(dir)
    os.makedirs(dir)


def main():
    random.seed(0)
    split_rate = 0.1

    cwd = os.getcwd()
    data_path = os.path.join(cwd, "flower_data")
    origin_data_path = "../dataset/flower_dataset"
    assert os.path.exists(origin_data_path), \
        "path {} does not exist.".format(origin_data_path)

    # os.listdir(path)      返回指定路径下的文件和文件夹列表
    # os.path.isdir(path)   判断路径是否为目录
    flower_class = [cla for cla in os.listdir(origin_data_path)
                    if os.path.isdir(os.path.join(origin_data_path, cla))]

    train_path = os.path.join(data_path, "train")
    mk_dir(train_path)
    for cla in flower_class:
        mk_dir(os.path.join(train_path, cla))

    valid_path = os.path.join(data_path, "valid")
    mk_dir(valid_path)
    for cla in flower_class:
        mk_dir(os.path.join(valid_path, cla))

    for cla in flower_class:
        valid_num, train_num = 0, 0
        cla_path = os.path.join(origin_data_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        valid_index = random.sample(range(num), k=int(num * split_rate))

        for index, image in enumerate(images):
            image_path = os.path.join(cla_path, image)
            if index in valid_index:
                copyfile(image_path, os.path.join(valid_path, cla, "{}.jpg".format(valid_num)))
                valid_num += 1
            else:
                copyfile(image_path, os.path.join(train_path, cla, "{}.jpg".format(train_num)))
                train_num += 1


if __name__ == '__main__':
    print("data prepare start!")
    main()
    print("data prepare done!")
```

* 导入数据集

```python
def data_loader_prepare(batch_size):
    # set transforms
    transforms_ = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    }

    # data set
    data_path = os.path.join(os.getcwd(), "flower_data")
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    train_dataset = datasets.ImageFolder(root=train_path, transform=transforms_["train"])
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=transforms_["valid"])

    # class dict
    class2index_dict = train_dataset.class_to_idx
    index2class_dict = dict((value, key) for key, value in class2index_dict.items())
    with open('index2class.json', 'w') as json_file:
        json.dump(index2class_dict, json_file)  # test : index2class_dict = json.load(json_file)
```

1. 利用 **datasets.ImageFolder** 导入图像数据集，默认已经按照类别分好了文件夹

2. 利用标准库 **json** 便于测试时直接读取

3. > class_to_idx (Dict[str, int]) – Dictionary mapping class name to class index

## tqdm

[tqdm](https://tqdm.github.io/)是一个进度管理器，但是使用过程中，踩过一些坑

* >TypeError: 'module' object is not callable

    ```python
    from tqdm import tqdm  # not import tqdm
    ```

* 无法在一行内显示进度条，PyCharm内多行滚动……

```python
    desc = "[EPOCH {:>3d} / {:>3d}] TRAIN".format(epoch+1, EPOCH)
    with tqdm(data_loader, desc=desc, ncols=80, file=sys.stdout) as train_bar:
        for (x, y) in train_bar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    print("[EPOCH {:>3d} / {:>3d}] TRAIN LOSS : {:.6f} ".format(epoch + 1, EPOCH, train_loss))
```

## tensorboard

了解标准库 **logging** 后对tensorboard有了更清晰的认识

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('log')
writer.add_scalar('Train Loss', train_loss, epoch + 1)
writer.close()
```

```shell
tensorboard --logdir=log
```
