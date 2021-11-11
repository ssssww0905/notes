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
