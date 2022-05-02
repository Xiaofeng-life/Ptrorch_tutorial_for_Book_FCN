import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, label, smooth=1):
        # 将预测结果pred和真实标签label进行展开操作
        # 展开后的pred和label均为一维的向量，便于计算交并比
        pred = F.sigmoid(pred)
        pred = pred.view(-1)
        label = label.view(-1)

        # 计算pred和label的交集
        intersection = (pred * label).sum()
        # 计算pred和label的并集
        union = pred.sum() + label.sum()
        # 根据前述的dice损失计算公式，加入平滑因子计算交并比
        dice = (2. * intersection + smooth) / (union + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                smooth: int = 1
                ) -> torch.Tensor:
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, label, alpha=0.8, gamma=2):
        pred = F.sigmoid(pred)
        # 将pred和label展开为一维
        pred = pred.view(-1)
        label = label.view(-1)

        # 计算BCE损失
        bce = F.binary_cross_entropy(pred, label, reduction='mean')
        bce = torch.exp(-bce)

        # Focal Loss的核心计算公式
        focal_loss = alpha * (1 - bce) ** gamma * bce
        return focal_loss


class LossWriter():
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def add(self, loss_name, loss, i):
        with open(os.path.join(self.save_dir, loss_name + ".txt"), mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()


import matplotlib.pyplot as plt



def plot_loss(txt_name, x_label, y_label, title, legend, font_size, fig_size, save_name):
    """
    损失函数绘图代码
    """
    all_i = []
    all_val = []
    with open(txt_name, "r") as f:
        # 读取txt文件中的所有行
        all_lines = f.readlines()
        # 遍历每一行
        for line in all_lines:
            # 每行的第一个元素和第二个元素以空格分隔
            sp= line.split(" ")
            i = int(sp[0])
            val = float(sp[1])
            all_i.append(i)
            all_val.append(val)
    # 绘图以及参数指定
    plt.figure(figsize=(6, 4))
    plt.plot(all_i, all_val)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    if legend:
        plt.legend(legend, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.savefig(save_name, dpi=200, bbox_inches = "tight")
    plt.show()


if __name__ == "__main__":
    plot_loss(txt_name="results_unet/loss/bce_loss.txt", x_label="iteration",
              y_label="loss value", title="Loss of BCE on UNet",
              legend=None, font_size=15, fig_size=(10, 10),
              save_name="unet_BCE_loss.png")