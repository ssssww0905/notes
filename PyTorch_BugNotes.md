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

```python

```

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
