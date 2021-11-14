# 记录一些踩过的坑 PyTorch

## nn.Sequential

* **问题描述**

```powershell
C:\Users\Administrator\anaconda3\envs\pytorch\python.exe D:/GitHub/get_start_with_pytorch/vgg/model.py
Traceback (most recent call last):
  File "D:/GitHub/get_start_with_pytorch/vgg/model.py", line 69, in <module>
    model = VGG("vgg_16")
  File "D:/GitHub/get_start_with_pytorch/vgg/model.py", line 36, in __init__
    self._features = nn.Sequential(features_seq)
  File "C:\Users\Administrator\anaconda3\envs\pytorch\lib\site-packages\torch\nn\modules\container.py", line 91, in __init__
    self.add_module(str(idx), module)
  File "C:\Users\Administrator\anaconda3\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 377, in add_module
    raise TypeError("{} is not a Module subclass".format(
TypeError: list is not a Module subclass
```

* **原因分析**

```python
features_seq = []
features_seq.append(nn.Conv2d(c_in, i, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
features_seq.append(nn.ReLU(inplace=True))
self._features = nn.Sequential(features_seq)
```

```python
class Sequential(Module):
    """
    Example::

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """
    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
```

* **解决方案**

```python
features_seq = []
features_seq.append(nn.Conv2d(c_in, i, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
features_seq.append(nn.ReLU(inplace=True))
self._features = nn.Sequential(*features_seq)  # 解包
```

## ResNet stride

* **问题描述**
`BasicBlock` `Bottleneck` **可能** 涉及 HW 的减半（conv2_x是通过maxpool完成的），所以在初始化时需要传入 **stride** 参数，且应该考虑具体在哪一个卷积层将高和宽变为原来的一半

* **代码参考**

```python
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
```

## ResNet _make_layer

* **问题描述**

**identity** 不能直接与 **out** 相加有两种可能：channel 不同或 HW 不同
channel 直接调整 **in_channels**  和 **out_channels**；HW 通过 **stride** 为 (1, 1) 或 (2, 2) 控制

* **代码参考**

```python
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

```

## transforms.resize

* **问题描述**

`resize` 的参数

* **代码参考**

```python
class Resize(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
    """
```

## load_state_dict

* **问题描述**

1. `load_state_dict` 网络中各层名字必须一致！【默认`strict=True`】

```python
model.load_state_dict(torch.load("{}_pretrained.pth".format(model.name), map_location=device), strict=True)
```

```powershell
RuntimeError: Error(s) in loading state_dict for ResNet:
 Missing key(s) in state_dict: "conv2.0.conv1.weight", "conv2.0.bn1.weight", "conv2.0.bn1.bias", "conv2.0.bn1.running_mean", "conv2.0.bn1.running_var", "conv2.0.conv2.weight", "conv2.0.bn2.weight", "conv2.0.bn2.bias", "conv2.0.bn2.running_mean", "conv2.0.bn2.running_var", "conv2.1.conv1.weight", "conv2.1.bn1.weight", "conv2.1.bn1.bias", "conv2.1.bn1.running_mean", "conv2.1.bn1.running_var", "conv2.1.conv2.weight", "conv2.1.bn2.weight", "conv2.1.bn2.bias", "conv2.1.bn2.running_mean", "conv2.1.bn2.running_var", "conv2.2.conv1.weight", "conv2.2.bn1.weight", "conv2.2.bn1.bias", "conv2.2.bn1.running_mean", "conv2.2.bn1.running_var", "conv2.2.conv2.weight", "conv2.2.bn2.weight", "conv2.2.bn2.bias", "conv2.2.bn2.running_mean", "conv2.2.bn2.running_var", "conv3.0.conv1.weight", "conv3.0.bn1.weight", "conv3.0.bn1.bias", "conv3.0.bn1.running_mean", "conv3.0.bn1.running_var", "conv3.0.conv2.weight", "conv3.0.bn2.weight", "conv3.0.bn2.bias", "conv3.0.bn2.running_mean", "conv3.0.bn2.running_var", "conv3.0.downsample.0.weight", "conv3.0.downsample.1.weight", "conv3.0.downsample.1.bias", "conv3.0.downsample.1.running_mean", "conv3.0.downsample.1.running_var", "conv3.1.conv1.weight", "conv3.1.bn1.weight", "conv3.1.bn1.bias", "conv3.1.bn1.running_mean", "conv3.1.bn1.running_var", "conv3.1.conv2.weight", "conv3.1.bn2.weight", "conv3.1.bn2.bias", "conv3.1.bn2.running_mean", "conv3.1.bn2.running_var", "conv3.2.conv1.weight", "conv3.2.bn1.weight", "conv3.2.bn1.bias", "conv3.2.bn1.running_mean", "conv3.2.bn1.running_var", "conv3.2.conv2.weight", "conv3.2.bn2.weight", "conv3.2.bn2.bias", "conv3.2.bn2.running_mean", "conv3.2.bn2.running_var", "conv3.3.conv1.weight", "conv3.3.bn1.weight", "conv3.3.bn1.bias", "conv3.3.bn1.running_mean", "conv3.3.bn1.running_var", "conv3.3.conv2.weight", "conv3.3.bn2.weight", "conv3.3.bn2.bias", "conv3.3.bn2.running_mean", "conv3.3.bn2.running_var", "conv4.0.conv1.weight", "conv4.0.bn1.weight", "conv4.0.bn1.bias", "conv4.0.bn1.running_mean", "conv4.0.bn1.running_var", "conv4.0.conv2.weight", "conv4.0.bn2.weight", "conv4.0.bn2.bias", "conv4.0.bn2.running_mean", "conv4.0.bn2.running_var", "conv4.0.downsample.0.weight", "conv4.0.downsample.1.weight", "conv4.0.downsample.1.bias", "conv4.0.downsample.1.running_mean", "conv4.0.downsample.1.running_var", "conv4.1.conv1.weight", "conv4.1.bn1.weight", "conv4.1.bn1.bias", "conv4.1.bn1.running_mean", "conv4.1.bn1.running_var", "conv4.1.conv2.weight", "conv4.1.bn2.weight", "conv4.1.bn2.bias", "conv4.1.bn2.running_mean", "conv4.1.bn2.running_var", "conv4.2.conv1.weight", "conv4.2.bn1.weight", "conv4.2.bn1.bias", "conv4.2.bn1.running_mean", "conv4.2.bn1.running_var", "conv4.2.conv2.weight", "conv4.2.bn2.weight", "conv4.2.bn2.bias", "conv4.2.bn2.running_mean", "conv4.2.bn2.running_var", "conv4.3.conv1.weight", "conv4.3.bn1.weight", "conv4.3.bn1.bias", "conv4.3.bn1.running_mean", "conv4.3.bn1.running_var", "conv4.3.conv2.weight", "conv4.3.bn2.weight", "conv4.3.bn2.bias", "conv4.3.bn2.running_mean", "conv4.3.bn2.running_var", "conv4.4.conv1.weight", "conv4.4.bn1.weight", "conv4.4.bn1.bias", "conv4.4.bn1.running_mean", "conv4.4.bn1.running_var", "conv4.4.conv2.weight", "conv4.4.bn2.weight", "conv4.4.bn2.bias", "conv4.4.bn2.running_mean", "conv4.4.bn2.running_var", "conv4.5.conv1.weight", "conv4.5.bn1.weight", "conv4.5.bn1.bias", "conv4.5.bn1.running_mean", "conv4.5.bn1.running_var", "conv4.5.conv2.weight", "conv4.5.bn2.weight", "conv4.5.bn2.bias", "conv4.5.bn2.running_mean", "conv4.5.bn2.running_var", "conv5.0.conv1.weight", "conv5.0.bn1.weight", "conv5.0.bn1.bias", "conv5.0.bn1.running_mean", "conv5.0.bn1.running_var", "conv5.0.conv2.weight", "conv5.0.bn2.weight", "conv5.0.bn2.bias", "conv5.0.bn2.running_mean", "conv5.0.bn2.running_var", "conv5.0.downsample.0.weight", "conv5.0.downsample.1.weight", "conv5.0.downsample.1.bias", "conv5.0.downsample.1.running_mean", "conv5.0.downsample.1.running_var", "conv5.1.conv1.weight", "conv5.1.bn1.weight", "conv5.1.bn1.bias", "conv5.1.bn1.running_mean", "conv5.1.bn1.running_var", "conv5.1.conv2.weight", "conv5.1.bn2.weight", "conv5.1.bn2.bias", "conv5.1.bn2.running_mean", "conv5.1.bn2.running_var", "conv5.2.conv1.weight", "conv5.2.bn1.weight", "conv5.2.bn1.bias", "conv5.2.bn1.running_mean", "conv5.2.bn1.running_var", "conv5.2.conv2.weight", "conv5.2.bn2.weight", "conv5.2.bn2.bias", "conv5.2.bn2.running_mean", "conv5.2.bn2.running_var". 
 Unexpected key(s) in state_dict: "layer1.0.conv1.weight", "layer1.0.bn1.running_mean", "layer1.0.bn1.running_var", "layer1.0.bn1.weight", "layer1.0.bn1.bias", "layer1.0.conv2.weight", "layer1.0.bn2.running_mean", "layer1.0.bn2.running_var", "layer1.0.bn2.weight", "layer1.0.bn2.bias", "layer1.1.conv1.weight", "layer1.1.bn1.running_mean", "layer1.1.bn1.running_var", "layer1.1.bn1.weight", "layer1.1.bn1.bias", "layer1.1.conv2.weight", "layer1.1.bn2.running_mean", "layer1.1.bn2.running_var", "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer1.2.conv1.weight", "layer1.2.bn1.running_mean", "layer1.2.bn1.running_var", "layer1.2.bn1.weight", "layer1.2.bn1.bias", "layer1.2.conv2.weight", "layer1.2.bn2.running_mean", "layer1.2.bn2.running_var", "layer1.2.bn2.weight", "layer1.2.bn2.bias", "layer2.0.conv1.weight", "layer2.0.bn1.running_mean", "layer2.0.bn1.running_var", "layer2.0.bn1.weight", "layer2.0.bn1.bias", "layer2.0.conv2.weight", "layer2.0.bn2.running_mean", "layer2.0.bn2.running_var", "layer2.0.bn2.weight", "layer2.0.bn2.bias", "layer2.0.downsample.0.weight", "layer2.0.downsample.1.running_mean", "layer2.0.downsample.1.running_var", "layer2.0.downsample.1.weight", "layer2.0.downsample.1.bias", "layer2.1.conv1.weight", "layer2.1.bn1.running_mean", "layer2.1.bn1.running_var", "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.conv2.weight", "layer2.1.bn2.running_mean", "layer2.1.bn2.running_var", "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer2.2.conv1.weight", "layer2.2.bn1.running_mean", "layer2.2.bn1.running_var", "layer2.2.bn1.weight", "layer2.2.bn1.bias", "layer2.2.conv2.weight", "layer2.2.bn2.running_mean", "layer2.2.bn2.running_var", "layer2.2.bn2.weight", "layer2.2.bn2.bias", "layer2.3.conv1.weight", "layer2.3.bn1.running_mean", "layer2.3.bn1.running_var", "layer2.3.bn1.weight", "layer2.3.bn1.bias", "layer2.3.conv2.weight", "layer2.3.bn2.running_mean", "layer2.3.bn2.running_var", "layer2.3.bn2.weight", "layer2.3.bn2.bias", "layer3.0.conv1.weight", "layer3.0.bn1.running_mean", "layer3.0.bn1.running_var", "layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.conv2.weight", "layer3.0.bn2.running_mean", "layer3.0.bn2.running_var", "layer3.0.bn2.weight", "layer3.0.bn2.bias", "layer3.0.downsample.0.weight", "layer3.0.downsample.1.running_mean", "layer3.0.downsample.1.running_var", "layer3.0.downsample.1.weight", "layer3.0.downsample.1.bias", "layer3.1.conv1.weight", "layer3.1.bn1.running_mean", "layer3.1.bn1.running_var", "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.conv2.weight", "layer3.1.bn2.running_mean", "layer3.1.bn2.running_var", "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer3.2.conv1.weight", "layer3.2.bn1.running_mean", "layer3.2.bn1.running_var", "layer3.2.bn1.weight", "layer3.2.bn1.bias", "layer3.2.conv2.weight", "layer3.2.bn2.running_mean", "layer3.2.bn2.running_var", "layer3.2.bn2.weight", "layer3.2.bn2.bias", "layer3.3.conv1.weight", "layer3.3.bn1.running_mean", "layer3.3.bn1.running_var", "layer3.3.bn1.weight", "layer3.3.bn1.bias", "layer3.3.conv2.weight", "layer3.3.bn2.running_mean", "layer3.3.bn2.running_var", "layer3.3.bn2.weight", "layer3.3.bn2.bias", "layer3.4.conv1.weight", "layer3.4.bn1.running_mean", "layer3.4.bn1.running_var", "layer3.4.bn1.weight", "layer3.4.bn1.bias", "layer3.4.conv2.weight", "layer3.4.bn2.running_mean", "layer3.4.bn2.running_var", "layer3.4.bn2.weight", "layer3.4.bn2.bias", "layer3.5.conv1.weight", "layer3.5.bn1.running_mean", "layer3.5.bn1.running_var", "layer3.5.bn1.weight", "layer3.5.bn1.bias", "layer3.5.conv2.weight", "layer3.5.bn2.running_mean", "layer3.5.bn2.running_var", "layer3.5.bn2.weight", "layer3.5.bn2.bias", "layer4.0.conv1.weight", "layer4.0.bn1.running_mean", "layer4.0.bn1.running_var", "layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.conv2.weight", "layer4.0.bn2.running_mean", "layer4.0.bn2.running_var", "layer4.0.bn2.weight", "layer4.0.bn2.bias", "layer4.0.downsample.0.weight", "layer4.0.downsample.1.running_mean", "layer4.0.downsample.1.running_var", "layer4.0.downsample.1.weight", "layer4.0.downsample.1.bias", "layer4.1.conv1.weight", "layer4.1.bn1.running_mean", "layer4.1.bn1.running_var", "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.conv2.weight", "layer4.1.bn2.running_mean", "layer4.1.bn2.running_var", "layer4.1.bn2.weight", "layer4.1.bn2.bias", "layer4.2.conv1.weight", "layer4.2.bn1.running_mean", "layer4.2.bn1.running_var", "layer4.2.bn1.weight", "layer4.2.bn1.bias", "layer4.2.conv2.weight", "layer4.2.bn2.running_mean", "layer4.2.bn2.running_var", "layer4.2.bn2.weight", "layer4.2.bn2.bias".

```

2. 分类问题数目不同，如何加载预训练的参数？

3. load中 **map_location** 参数，以及 `model.to(device)` 的位置？

* **解决方案**

```python
model = resnet50(class_num=1000, include_top=True)
model.load_state_dict(torch.load("{}_pretrained.pth".format(model.name), map_location=device), strict=False)
# change fc layer to match this classification
model.fc = nn.Linear(model.fc.in_features, 5)

model.to(device)
```

* **代码参考**

1. `state_dict`

<details>
<summary>model.state_dict()</summary>

```python
model = resnet34()
print(type(model.state_dict()))
print(model.state_dict().keys())
```

```powershell
<class 'collections.OrderedDict'>
odict_keys(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bi
as', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.run
ning_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.
bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'lay
er1.1.bn2.num_batches_tracked', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.num_batches_
tracked', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked', 'layer2.0.co
nv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight', 'layer2.0
.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.w
eight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight', '
layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight',
'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', '
layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_m
ean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean', 'layer2.3.bn1.ru
nning_var', 'layer2.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.
bn2.num_batches_tracked', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracke
d', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsamp
le.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batche
s_tracked', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.
conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer3.2.conv1.weight', 'layer3
.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer
3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias', 'layer3
.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean',
'layer3.3.bn2.running_var', 'layer3.3.bn2.num_batches_tracked', 'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_
var', 'layer3.4.bn1.num_batches_tracked', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.bn2.nu
m_batches_tracked', 'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var', 'layer3.5.bn1.num_batches_tracked', 'l
ayer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer3.5.bn2.num_batches_tracked', 'layer4.0.conv1.weight',
 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight'
, 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'lay
er4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn1
.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn
2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1
.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.num_batches_tracked', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer
4.2.bn2.running_var', 'layer4.2.bn2.num_batches_tracked', 'fc.weight', 'fc.bias'])
```

</details>

<details>
<summary>optimizer.state_dict()</summary>

```python
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01)  # model.parameters()
print(type(optimizer.state_dict()))
print(optimizer.state_dict().keys())
print(optimizer.state_dict().values())
print(type(optimizer.state_dict()['param_groups']))
print(type(optimizer.state_dict()['param_groups'][0]))
```

```powershell
<class 'dict'>
dict_keys(['state', 'param_groups'])
dict_values([{}, [{'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}]
])
<class 'list'>
<class 'dict'>
```

</details>

2. `torch.load`

>If :attr:`map_location` is a :class:`torch.device` object or a string containing a device tag, it indicates the location where all tensors should be loaded.

<details>
<summary>torch.load()</summary>

```python
def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """load(f, map_location=None, pickle_module=pickle, **pickle_load_args)

    Loads an object saved with :func:`torch.save` from a file.

    :func:`torch.load` uses Python's unpickling facilities but treats storages,
    which underlie tensors, specially. They are first deserialized on the
    CPU and are then moved to the device they were saved from. If this fails
    (e.g. because the run time system doesn't have certain devices), an exception
    is raised. However, storages can be dynamically remapped to an alternative
    set of devices using the :attr:`map_location` argument.

    If :attr:`map_location` is a callable, it will be called once for each serialized
    storage with two arguments: storage and location. The storage argument
    will be the initial deserialization of the storage, residing on the CPU.
    Each serialized storage has a location tag associated with it which
    identifies the device it was saved from, and this tag is the second
    argument passed to :attr:`map_location`. The builtin location tags are ``'cpu'``
    for CPU tensors and ``'cuda:device_id'`` (e.g. ``'cuda:2'``) for CUDA tensors.
    :attr:`map_location` should return either ``None`` or a storage. If
    :attr:`map_location` returns a storage, it will be used as the final deserialized
    object, already moved to the right device. Otherwise, :func:`torch.load` will
    fall back to the default behavior, as if :attr:`map_location` wasn't specified.

    If :attr:`map_location` is a :class:`torch.device` object or a string containing
    a device tag, it indicates the location where all tensors should be loaded.

    Otherwise, if :attr:`map_location` is a dict, it will be used to remap location tags
    appearing in the file (keys), to ones that specify where to put the
    storages (values).

    User extensions can register their own location tags and tagging and
    deserialization methods using :func:`torch.serialization.register_package`.

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
            or a string or os.PathLike object containing a file name
        map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations
        pickle_module: module used for unpickling metadata and objects (has to
            match the :attr:`pickle_module` used to serialize file)
        pickle_load_args: (Python 3 only) optional keyword arguments passed over to
            :func:`pickle_module.load` and :func:`pickle_module.Unpickler`, e.g.,
            :attr:`errors=...`.

    .. warning::
        :func:`torch.load()` uses ``pickle`` module implicitly, which is known to be insecure.
        It is possible to construct malicious pickle data which will execute arbitrary code
        during unpickling. Never load data that could have come from an untrusted
        source, or that could have been tampered with. **Only load data you trust**.

    .. note::
        When you call :func:`torch.load()` on a file which contains GPU tensors, those tensors
        will be loaded to GPU by default. You can call ``torch.load(.., map_location='cpu')``
        and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.

    .. note::
        By default, we decode byte strings as ``utf-8``.  This is to avoid a common error
        case ``UnicodeDecodeError: 'ascii' codec can't decode byte 0x...``
        when loading files saved by Python 2 in Python 3.  If this default
        is incorrect, you may use an extra :attr:`encoding` keyword argument to specify how
        these objects should be loaded, e.g., :attr:`encoding='latin1'` decodes them
        to strings using ``latin1`` encoding, and :attr:`encoding='bytes'` keeps them
        as byte arrays which can be decoded later with ``byte_array.decode(...)``.

    Example:
        >>> torch.load('tensors.pt')
        # Load all tensors onto the CPU
        >>> torch.load('tensors.pt', map_location=torch.device('cpu'))
        # Load all tensors onto the CPU, using a function
        >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
        # Load all tensors onto GPU 1
        >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
        # Map tensors from GPU 1 to GPU 0
        >>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
        # Load tensor from io.BytesIO object
        >>> with open('tensor.pt', 'rb') as f:
        ...     buffer = io.BytesIO(f.read())
        >>> torch.load(buffer)
        # Load a module with 'ascii' encoding for unpickling
        >>> torch.load('module.pt', encoding='ascii')
    """
    _check_dill_version(pickle_module)

    if 'encoding' not in pickle_load_args.keys():
        pickle_load_args['encoding'] = 'utf-8'

    with _open_file_like(f, 'rb') as opened_file:
        if _is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            orig_position = opened_file.tell()
            with _open_zipfile_reader(opened_file) as opened_zipfile:
                if _is_torchscript_zip(opened_zipfile):
                    warnings.warn("'torch.load' received a zip file that looks like a TorchScript archive"
                                  " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to"
                                  " silence this warning)", UserWarning)
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file)
                return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
        return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
```

</details>

3. `load_state_dict`

<details>
<summary>load_state_dict()</summary>

```python
def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                    strict: bool = True):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Note:
        If a parameter or buffer is registered as ``None`` and its corresponding key
        exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
        ``RuntimeError``.
    """
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(self)
    del load

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                            self.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)
```

</details>

4. `.to(device)`

<details>
<summary>Tensor.to()</summary>

```python
def to(self, *args, **kwargs): # real signature unknown; restored from __doc__
    """
    to(*args, **kwargs) -> Tensor
    
    Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
    inferred from the arguments of ``self.to(*args, **kwargs)``.
    
    .. note::
    
        If the ``self`` Tensor already
        has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
        Otherwise, the returned tensor is a copy of ``self`` with the desired
        :class:`torch.dtype` and :class:`torch.device`.
    
    Here are the ways to call ``to``:
    
    .. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
        :noindex:
    
        Returns a Tensor with the specified :attr:`dtype`
    
        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.preserve_format``.
    
    .. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
        :noindex:
    
        Returns a Tensor with the specified :attr:`device` and (optional)
        :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
        When :attr:`non_blocking`, tries to convert asynchronously with respect to
        the host if possible, e.g., converting a CPU Tensor with pinned memory to a
        CUDA Tensor.
        When :attr:`copy` is set, a new Tensor is created even when the Tensor
        already matches the desired conversion.
    
        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.preserve_format``.
    
    .. method:: to(other, non_blocking=False, copy=False) -> Tensor
        :noindex:
    
        Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
        the Tensor :attr:`other`. When :attr:`non_blocking`, tries to convert
        asynchronously with respect to the host if possible, e.g., converting a CPU
        Tensor with pinned memory to a CUDA Tensor.
        When :attr:`copy` is set, a new Tensor is created even when the Tensor
        already matches the desired conversion.
    
    Example::
    
        >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
        >>> tensor.to(torch.float64)
        tensor([[-0.5044,  0.0005],
                [ 0.3310, -0.0584]], dtype=torch.float64)
    
        >>> cuda0 = torch.device('cuda:0')
        >>> tensor.to(cuda0)
        tensor([[-0.5044,  0.0005],
                [ 0.3310, -0.0584]], device='cuda:0')
    
        >>> tensor.to(cuda0, dtype=torch.float64)
        tensor([[-0.5044,  0.0005],
                [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
    
        >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
        >>> tensor.to(other, non_blocking=True)
        tensor([[-0.5044,  0.0005],
                [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
    """
    return _te.Tensor(*(), **{})
```

</details>

<details>
<summary>Module.to()</summary>

```python
def to(self, *args, **kwargs):
    r"""Moves and/or casts the parameters and buffers.

    This can be called as

    .. function:: to(device=None, dtype=None, non_blocking=False)
        :noindex:

    .. function:: to(dtype, non_blocking=False)
        :noindex:

    .. function:: to(tensor, non_blocking=False)
        :noindex:

    .. function:: to(memory_format=torch.channels_last)
        :noindex:

    Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
    floating point or complex :attr:`dtype`\ s. In addition, this method will
    only cast the floating point or complex parameters and buffers to :attr:`dtype`
    (if given). The integral parameters and buffers will be moved
    :attr:`device`, if that is given, but with dtypes unchanged. When
    :attr:`non_blocking` is set, it tries to convert/move asynchronously
    with respect to the host if possible, e.g., moving CPU Tensors with
    pinned memory to CUDA devices.

    See below for examples.

    .. note::
        This method modifies the module in-place.

    Args:
        device (:class:`torch.device`): the desired device of the parameters
            and buffers in this module
        dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
            the parameters and buffers in this module
        tensor (torch.Tensor): Tensor whose dtype and device are the desired
            dtype and device for all parameters and buffers in this module
        memory_format (:class:`torch.memory_format`): the desired memory
            format for 4D parameters and buffers in this module (keyword
            only argument)

    Returns:
        Module: self

    Examples::

        >>> linear = nn.Linear(2, 2)
        >>> linear.weight
        Parameter containing:
        tensor([[ 0.1913, -0.3420],
                [-0.5113, -0.2325]])
        >>> linear.to(torch.double)
        Linear(in_features=2, out_features=2, bias=True)
        >>> linear.weight
        Parameter containing:
        tensor([[ 0.1913, -0.3420],
                [-0.5113, -0.2325]], dtype=torch.float64)
        >>> gpu1 = torch.device("cuda:1")
        >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
        Linear(in_features=2, out_features=2, bias=True)
        >>> linear.weight
        Parameter containing:
        tensor([[ 0.1914, -0.3420],
                [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
        >>> cpu = torch.device("cpu")
        >>> linear.to(cpu)
        Linear(in_features=2, out_features=2, bias=True)
        >>> linear.weight
        Parameter containing:
        tensor([[ 0.1914, -0.3420],
                [-0.5112, -0.2324]], dtype=torch.float16)

        >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
        >>> linear.weight
        Parameter containing:
        tensor([[ 0.3741+0.j,  0.2382+0.j],
                [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
        >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
        tensor([[0.6122+0.j, 0.1150+0.j],
                [0.6122+0.j, 0.1150+0.j],
                [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

    """

    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

    if dtype is not None:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError('nn.Module.to only accepts floating point or complex '
                            'dtypes, but got desired dtype={}'.format(dtype))
        if dtype.is_complex:
            warnings.warn(
                "Complex modules are a new feature under active development whose design may change, "
                "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.md "
                "if a complex module does not work as expected.")

    def convert(t):
        if convert_to_format is not None and t.dim() in (4, 5):
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

    return self._apply(convert)
```

</details>

## json

* **问题描述**

注意 `json.load()` 与 `json.dump()` 的用法，总结来说就是 **得先打开文件**

* **代码参考**

```python
with open('index2class.json', 'w') as json_file:
    json.dump(index2class_dict, json_file)

with open('index2class.json', 'r') as json_file:
    index2class = json.load(json_file)
```

## torch.softmax

* **问题描述**

```python
pred_prob = torch.softmax(pred)
```

```powershell
TypeError: softmax() received an invalid combination of arguments - got (Tensor), but expected one of:
 * (Tensor input, int dim, torch.dtype dtype)
 * (Tensor input, name dim, *, torch.dtype dtype)
```

* **解决方案**

在预测时要注意图片的维度

```python
model.eval()
with torch.no_grad():
    img = img.to(device)  # (1, 3, 224, 224)
    pred = torch.squeeze(model(img))  # (5,)

    pred_prob = torch.softmax(pred, dim=0).cpu()
    pred_index = torch.argmax(pred_prob).numpy()
################################################
model.eval()
with torch.no_grad():
    img = img.to(device)  # (1, 3, 224, 224)
    pred = model(img)  # (1, 5)

    pred_prob = torch.softmax(pred, dim=1).cpu()
    pred_index = torch.argmax(pred_prob).numpy()
```
