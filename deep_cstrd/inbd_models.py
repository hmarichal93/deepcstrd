"""
Backboned U-Net for inbd segmentation: https://github.com/alexander-g/INBD/blob/master/src/models.py
"""
import typing as tp
import torch, torchvision
from torchvision.models._utils import IntermediateLayerGetter


MODULES = []

class UNet(torch.nn.Module):
    '''Backboned U-Net'''

    class UpBlock(torch.nn.Module):
        def __init__(self, in_c, out_c, inter_c=None):
            #super().__init__()
            torch.nn.Module.__init__(self)
            inter_c        = inter_c or out_c
            self.conv1x1   = torch.nn.Conv2d(in_c, inter_c, 1)
            self.convblock = torch.nn.Sequential(
                torch.nn.Conv2d(inter_c, out_c, 3, padding=1, bias=0),
                torch.nn.BatchNorm2d(out_c),
                torch.nn.ReLU(),
            )
        def forward(self, x:torch.Tensor, skip_x:torch.Tensor, relu=True) -> torch.Tensor:
            x = torch.nn.functional.interpolate(x, skip_x.shape[2:])   #TODO? mode='bilinear
            x = torch.cat([x, skip_x], dim=1)
            x = self.conv1x1(x)
            x = self.convblock(x)
            return x
    
    def __init__(self, backbone='mobilenet3l', out_channels=1, downsample_factor=1, backbone_pretrained:bool=True):
        torch.nn.Module.__init__(self)
        factory_func = BACKBONES.get(backbone, None)
        if factory_func is None:
            raise NotImplementedError(backbone)
        self.backbone, C = factory_func(backbone_pretrained)
        self.backbone_name = backbone
        self.scale       = downsample_factor
        
        self.up0 = self.UpBlock(C[-1]    + C[-2],  C[-2])
        self.up1 = self.UpBlock(C[-2]    + C[-3],  C[-3])
        self.up2 = self.UpBlock(C[-3]    + C[-4],  C[-4])
        self.up3 = self.UpBlock(C[-4]    + C[-5],  C[-5])
        self.up4 = self.UpBlock(C[-5]    + 3,      32)
        self.cls = torch.nn.Conv2d(32, out_channels, 3, padding=1)
    
    def forward(self, x:torch.Tensor, sigmoid=False, return_features=False) -> torch.Tensor:
        device = list(self.parameters())[0].device
        x      = x.to(device)
        # Check input size
        if (x.shape[-2] < 5 or x.shape[-1] < 5):
            return torch.zeros((x.shape[0], self.cls.out_channels, x.shape[-2], x.shape[-1]), device=device)

        X = self.backbone(x)
        X = ([x] + [X[f'out{i}'] for i in range(5)])[::-1]
        x = X.pop(0)
        x = self.up0(x, X[0])
        x = self.up1(x, X[1])
        x = self.up2(x, X[2])
        x = self.up3(x, X[3])
        x = self.up4(x, X[4])
        if return_features:
            return x
        x = self.cls(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x
    

    


def resnet18_backbone(pretrained:bool) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.resnet18(pretrained=pretrained)
    return_layers = dict(relu='out0', layer1='out1', layer2='out2', layer3='out3', layer4='out4')
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [64, 64, 128, 256, 512]
    return backbone, channels

def resnet50_backbone(pretrained:bool) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.resnet50(pretrained=pretrained)
    return_layers = dict(relu='out0', layer1='out1', layer2='out2', layer3='out3', layer4='out4')
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [64, 256, 512, 1024, 2048]
    return backbone, channels

def mobilenet3l_backbone(pretrained:bool) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
    return_layers = {'1':'out0', '3':'out1', '6':'out2', '10':'out3', '16':'out4'}
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [16, 24, 40, 80, 960]
    return backbone, channels

BACKBONES = {
    'resnet18':    resnet18_backbone,
    'resnet50':    resnet50_backbone,
    'mobilenet3l': mobilenet3l_backbone,
}
