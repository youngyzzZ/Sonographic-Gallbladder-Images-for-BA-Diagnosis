# @ FileName: net.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
from torch.utils import model_zoo
from torchvision.models import resnet
from src.net_base import *
import torch.nn.functional as F
import pretrainedmodels
import torchvision
from efficientnet_pytorch import EfficientNet

NUM_BLOCKS = {
    18: [2, 2, 2, 2],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}


class Res18(resnet.ResNet):
    def __init__(self, num_classes=10):
        super(Res18, self).__init__(resnet.BasicBlock, NUM_BLOCKS[18])
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        init_params(self.fc)


class Res50(resnet.ResNet):
    def __init__(self, num_classes=10):
        super(Res50, self).__init__(resnet.Bottleneck, NUM_BLOCKS[50])
        self.expansion = resnet.Bottleneck.expansion
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        init_params(self.fc)


class Res101(resnet.ResNet):
    def __init__(self, num_classes=10):
        super(Res101, self).__init__(resnet.Bottleneck, NUM_BLOCKS[101])
        self.expansion = resnet.Bottleneck.expansion
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet101']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        init_params(self.fc)


class Res152(resnet.ResNet):
    def __init__(self, num_classes=10):
        super(Res152, self).__init__(resnet.Bottleneck, NUM_BLOCKS[152])
        self.expansion = resnet.Bottleneck.expansion
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet152']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        init_params(self.fc)


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()

        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )

        # load pre_train model
        pre_model = pretrainedmodels.inceptionv4(num_classes=1000,
                                                 pretrained='imagenet')
        self.last_linear = pre_model.last_linear
        self.load_state_dict(pre_model.state_dict())

        self.last_linear = nn.Linear(1536, num_classes)
        init_params(self.last_linear)

    def logits(self, features):
        # Allows image of any size to be processed
        if isinstance(features.shape[2], int):
            adaptive_avg_pool_width = features.shape[2]
        else:
            adaptive_avg_pool_width = features.shape[2].item()
        x = F.avg_pool2d(features, kernel_size=adaptive_avg_pool_width)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = self._make_layer(Block35, scale=0.17, blocks=10)
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = self._make_layer(Block17, scale=0.10, blocks=20)
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = self._make_layer(Block8, scale=0.20, blocks=9)
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)

        # load pre_train model
        pre_model = pretrainedmodels.inceptionresnetv2(num_classes=1000,
                                                       pretrained='imagenet')
        self.last_linear = pre_model.last_linear
        self.load_state_dict(pre_model.state_dict())

        self.last_linear = nn.Linear(1536, num_classes)
        init_params(self.last_linear)

    def _make_layer(self, block, scale, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(scale=scale))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class PnasNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(PnasNet, self).__init__()
        self.features = pretrainedmodels.pnasnet5large(num_classes=1000, pretrained='imagenet')
        self.features.last_linear = nn.Linear(4320, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x


class Se_resnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Se_resnet, self).__init__()
        self.features = pretrainedmodels.se_resnet152(num_classes=1000, pretrained='imagenet')
        self.features.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x


class Densnet201(nn.Module):
    def __init__(self, num_classes=1000):
        super(Densnet201, self).__init__()
        self.features = torchvision.models.densenet201(pretrained=True)
        self.features.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x


class Efficientnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Efficientnet, self).__init__()
        self.features = EfficientNet.from_pretrained('efficientnet-b7')
        self.features._fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x


def get_net(args):
    model = args['model']
    num_classes = args['num_classes']

    if model == 'res18':
        return Res18(num_classes=num_classes)
    elif model == 'res50':
        return Res50(num_classes=num_classes)
    elif model == 'res101':
        return Res101(num_classes=num_classes)
    elif model == 'res152':
        return Res152(num_classes=num_classes)
    elif model == 'Inc-v4':
        return InceptionV4(num_classes=num_classes)
    elif model == 'IncRes-v2':
        return InceptionResNetV2(num_classes=num_classes)
    elif model == 'pnasnet':
        return PnasNet(num_classes=num_classes)
    elif model == 'se_resnet':
        return Se_resnet(num_classes=num_classes)
    elif model == 'densenet':
        return Densnet201(num_classes=num_classes)
    elif model == 'efficientnet':
        return Efficientnet(num_classes=num_classes)
    else:
        raise ValueError('No model: {}'.format(model))
