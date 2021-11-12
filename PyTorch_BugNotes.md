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
