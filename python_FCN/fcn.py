import torch
import torch.nn as nn
from torchvision import models


class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 获取基础网络resnet34
        pretrained_net = models.resnet34(pretrained=False)

        # 获取三个池化层
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]

        # 获得单通道的预测输出，计算三个尺度的分数
        self.scores1 = nn.Conv2d(128, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(512, num_classes, 1)

        # 8倍上采样
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)

        # 2倍上采样
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 获取1/8池化输出

        x = self.stage2(x)
        s2 = x  # 获取1/16池化输出

        x = self.stage3(x)
        s3 = x  # 获取1/32池化输出

        # 计算1/32分数
        s3_scores = self.scores3(s3)
        # 上采样2倍，并进行融合
        s3_scores_x2 = self.upsample_2x(s3_scores)
        s2_scores = self.scores2(s2)
        s2_fuse = s2_scores + s3_scores_x2

        # 计算1/8分数
        s1_scores = self.scores1(s1)
        # 上采样2倍，并进行融合
        s2_fuse_x2 = self.upsample_2x(s2_fuse)
        s = s1_scores + s2_fuse_x2

        # 上采样8倍，获取的s_x8与原始输入尺寸相同
        s_x8 = self.upsample_8x(s)
        s_x8 = self.sigmoid(s_x8)

        return s_x8


if __name__ == "__main__":
    fcn = FCN(num_classes=1)
    x = torch.randn(size=(1, 3, 224, 224))
    out = fcn(x)
    print(out.size())