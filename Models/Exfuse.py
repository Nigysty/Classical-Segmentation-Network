from PIL import Image
import torch.nn as nn
from torchvision import models
import torch

resnet101 = models.resnet101(pretrained=False)

class SEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x1, x2 = x
        return x1 * self.upsample(self.conv(x2))

class ECRE(nn.Module):
    def __init__(self, in_c, up_scale=2):
        super(ECRE, self).__init__()
        self.ecre = nn.Sequential(nn.Conv2d(in_c, in_c * up_scale * up_scale, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(in_c * up_scale * up_scale),
                                  nn.PixelShuffle(up_scale))

    def forward(self, input_):
        return self.ecre(input_)

class _GlobalConvModule(nn.Module):
    def __init__(self, in_channels, num_class, k=15):
        super(_GlobalConvModule, self).__init__()

        pad = (k-1) // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, num_class, kernel_size=(1, k), padding=(0, pad), bias=False),
                                   nn.Conv2d(num_class, num_class, kernel_size=(k, 1), padding=(pad, 0), bias=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, num_class, kernel_size=(k, 1), padding=(pad, 0), bias=False),
                                   nn.Conv2d(num_class, num_class, kernel_size=(1, k), padding=(0, pad), bias=False))

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        assert x1.shape == x2.shape

        return x1 + x2

class GCNFuse(nn.Module):
    def __init__(self, num_classes=21):
        super(GCNFuse, self).__init__()
        self.num_classes = num_classes
        dap_k = 3    # GCN Model中K的值
        
        self.resnet_features = models.resnet101(pretrained=False)
        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.resnet_features.bn1, self.resnet_features.relu)
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4

        self.gcm1 = _GlobalConvModule(2048, num_classes * 4)
        self.gcm2 = _GlobalConvModule(1024, num_classes)
        self.gcm3 = _GlobalConvModule(512, num_classes * dap_k**2)
        self.gcm4 = _GlobalConvModule(256, num_classes * dap_k**2)

        self.deconv1 = ECRE(num_classes)
        # self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(num_classes * dap_k**2, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(num_classes * dap_k**2, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(num_classes * dap_k**2, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)

        self.ecre = nn.PixelShuffle(2)

        self.seb1 = SEB(2048, 1024)
        self.seb2 = SEB(3072, 512)
        self.seb3 = SEB(3584, 256)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.DAP = nn.Sequential(
            nn.PixelShuffle(dap_k),
            nn.AvgPool2d((dap_k,dap_k))
        )

    def forward(self, x):
        # suppose input = x , if x 512
        f0 = self.layer0(x)  # 256
        f1 = self.layer1(f0)  # 128
        print (f1.size())
        f2 = self.layer2(f1)  # 64
        print (f2.size())
        f3 = self.layer3(f2)  # 32
        print (f3.size())
        f4 = self.layer4(f3)  # 16
        print (f4.size())
        x = self.gcm1(f4)
        out1 = self.ecre(x)
        seb1 = self.seb1([f3, f4])
        gcn1 = self.gcm2(seb1)

        seb2 = self.seb2([f2, torch.cat([f3, self.upsample2(f4)], dim=1)])
        gcn2 = self.gcm3(seb2)

        seb3 = self.seb3([f1, torch.cat([f2, self.upsample2(f3), self.upsample4(f4)], dim=1)])
        gcn3 = self.gcm4(seb3)

        y = self.deconv2(gcn1 + out1)
        y = self.deconv3(gcn2 + y)
        y = self.deconv4(gcn3 + y)
        y = self.deconv5(y)
        y = self.DAP(y)
        return y

if __name__ == '__main__':
    model = GCNFuse(21)
    model.eval()
    image = torch.randn(1, 3, 32, 32)
    res1 = model(image)
    print('result:', res1.size())