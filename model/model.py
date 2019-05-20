import torch.nn as nn
import torch.nn.functional as F
from model.attn_densenet import _AttnDenseNet
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class AttnDenseNet(BaseModel):
    "Thin wrapper over the actual Attention Densenet"

    def __init__(self,
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

        super(AttnDenseNet, self).__init__()
        self.attn_densenet = _AttnDenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,
            num_classes=num_classes,
            num_heads=num_heads,
            total_key_filters=total_key_filters,
            total_value_filters=total_value_filters,
            output_filters=output_filters
        )

    def forward(self, x):
        return self.attn_densenet(x)


if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    num_classes = 3

    net = AttnDenseNet(growth_rate=32,
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
    print(y.shape)
    import pkgutil
    search_path = ['.'] # set to None to see all modules importable from sys.path
    all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
    print(all_modules)

    import sys
    print(sys.path)