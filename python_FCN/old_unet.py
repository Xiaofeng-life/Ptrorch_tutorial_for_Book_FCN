import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, ch_in=3, ch_out=1):
        super().__init__()
        feature_channels = [8, 16, 32, 64, 128]

        # 定义四个池化层，池化核尺寸均为2，步长均为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 根据通道设定，定义下采样过程中使用的五个卷积层
        self.Conv1 = ConvBlock(ch_in, feature_channels[0])
        self.Conv2 = ConvBlock(feature_channels[0], feature_channels[1])
        self.Conv3 = ConvBlock(feature_channels[1], feature_channels[2])
        self.Conv4 = ConvBlock(feature_channels[2], feature_channels[3])
        self.Conv5 = ConvBlock(feature_channels[3], feature_channels[4])

        # 根据通道设定，定义上次样过程中使用的卷积层和上采样层
        self.Up5 = UpConvBlock(feature_channels[4], feature_channels[3])
        self.UpConv5 = ConvBlock(feature_channels[4], feature_channels[3])
        self.Up4 = UpConvBlock(feature_channels[3], feature_channels[2])
        self.UpConv4 = ConvBlock(feature_channels[3], feature_channels[2])
        self.Up3 = UpConvBlock(feature_channels[2], feature_channels[1])
        self.UpConv3 = ConvBlock(feature_channels[2], feature_channels[1])
        self.Up2 = UpConvBlock(feature_channels[1], feature_channels[0])
        self.UpConv2 = ConvBlock(feature_channels[1], feature_channels[0])

        # 定义最后一层卷积，输出通道数等于目标类别数
        self.Conv = nn.Conv2d(feature_channels[0], ch_out, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一层卷积，将3通道的输入变换为特征图
        f1 = self.Conv1(x)

        # 计算下采样过程，通过池化和卷积获得中间特征图
        f2 = self.pool1(f1)
        f2 = self.Conv2(f2)
        f3 = self.pool2(f2)
        f3 = self.Conv3(f3)
        f4 = self.pool3(f3)
        f4 = self.Conv4(f4)
        f5 = self.pool4(f4)
        f5 = self.Conv5(f5)

        # 第一次特征融合
        up_f5 = self.Up5(f5)
        up_f5 = torch.cat((f4, up_f5), dim=1)
        up_f5 = self.UpConv5(up_f5)

        # 第二次特征融合
        up_f4 = self.Up4(up_f5)
        up_f4 = torch.cat((f3, up_f4), dim=1)
        up_f4 = self.UpConv4(up_f4)

        # 第三次特征融合
        up_f3 = self.Up3(up_f4)
        up_f3 = torch.cat((f2, up_f3), dim=1)
        up_f3 = self.UpConv3(up_f3)

        # 第四次特征融合
        up_f2 = self.Up2(up_f3)
        up_f2 = torch.cat((f1, up_f2), dim=1)
        up_f2 = self.UpConv2(up_f2)

        # 计算最后一层卷积输出，获得预测mask
        mask = self.Conv(up_f2)
        mask = self.sigmoid(mask)
        return mask


if __name__ == "__main__":
    unet = UNet()
    x = torch.randn(size=(1, 3, 256, 256))
    print(unet(x).size())
    # torch.onnx.export(model=unet, args=x,
    #                   f="unet.onnx", input_names=["input"],
    #                   output_names=["output"], opset_version=11)
