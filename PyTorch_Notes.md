# PyTorch笔记
## [B, C, H, W]
* flatten
```
# 两种方式
x = torch.flatten(x, start_dim=1)  # dim_0 : batch_size
x = nn.Flatten()(x)  # default first dim to flatten = 1
```
* totensor
```
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
```
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
```
x_torch = torch.from_numpy(x_array)
```
* array2torch
```
x_array = x_torch.numpy()
```
## train
* get device 
```
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = x.to(device)
model = model.to(device)
```
* model.train() v.s. model.eval()
```
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

## model
* print model
```
print(net)
```
实验发现，net 架构只与 __init__ 中定义的self.layer有关，即使一个层在forward中使用多次，也只会打印一次
* print parameters
```
# named_parameters
for name, param in net.named_parameters():
	print("name: {:<15}, shape: {}".format(name, param.shape))

# parameters
for param in net.parameters():
	print("shape: {}".format(param.shape))
```
查看源码发现，net.parameters() 就是调用的 net.named_parameters()，所以以后还是用后者吧