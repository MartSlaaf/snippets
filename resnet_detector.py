from torchvision.models.resnet import ResNet
from torch import nn

class ResNetDetector(ResNet):
    """
    Detector Network.
    Full the same as usual ResNet, but will work as sliding window detector.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fc_s = self.fc.weight.shape
        self.final_conv = nn.Conv2d(fc_s[1], fc_s[0], 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.final_conv(x)

        return torch.softmax(x, 1)

from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

resnet_packs = {18: (resnet18, [BasicBlock, [2, 2, 2, 2]], {}),
                34: (resnet34, [BasicBlock, [3, 4, 6, 3]], {}),
                50: (resnet50, [Bottleneck, [3, 4, 6, 3]], {}),
                101: (resnet101, [Bottleneck, [3, 4, 23, 3]], {}),
                152: (resnet152, [Bottleneck, [3, 8, 36, 3]], {})}

def load_detector(resnet_model=18):
    """
    Load ResNet, pretrained on imagenet, and reformat it for detection task
    """
    load_origin, args, kwargs = resnet_packs[resnet_model]

    model = load_origin(pretrained=True)

    tmp_state_dict = model.state_dict()
    tmp_state_dict['final_conv.weight'] = tmp_state_dict['fc.weight'].unsqueeze(-1).unsqueeze(-1)
    tmp_state_dict['final_conv.bias'] = tmp_state_dict['fc.bias']

    model = ResNetDetector(*args, **kwargs)
    model.load_state_dict(tmp_state_dict)
    model.eval()

    return model
