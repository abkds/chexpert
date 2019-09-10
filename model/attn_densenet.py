import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from model.densenet import DenseBlock, _Transition
from model.attention import MultiHeadAttention


class _AttnDenseNet(nn.Module):
    r"""Densenet-BC + Attention model class, based on

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
            self,
            growth_rate=32,
            block_config=(
                6,
                12,
                24,
                16),
            num_init_features=64,
            bn_size=4,
            drop_rate=0,
            num_classes=1000,
            num_heads=8,
            total_key_filters=32,
            total_value_filters=32,
            output_filters=32):

        super(_AttnDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                num_heads=num_heads,
                total_key_filters=total_key_filters,
                total_value_filters=total_value_filters,
                output_filters=output_filters)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate + output_filters
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def total_params(model):
    return sum(p.numel() for p in model.parameters())

def get_img(path):
    X = io.imread(path)
    X.resize((*X.shape, 1))
    tsfrm = Compose([ToPILImage(), Resize((160, 320)), ToTensor()])
    return tsfrm(X)
    

# sample random input
if __name__ == '__main__':
    x = torch.rand(2, 1, 160, 160)
    num_classes = 3
    
    net = _AttnDenseNet(growth_rate=32,
                       block_config=(
                           6,
                           12,
                           24,
                           16),
                       num_init_features=64,
                       bn_size=4,
                       drop_rate=0,
                       num_classes=3,
                       num_heads=4,
                       total_key_filters=16,
                       total_value_filters=16,
                       output_filters=8)
    print(total_params(net))
    y = net(x)
    print(y)
    print(y.shape)
