#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/11 11:33
# @Author  : CongXiaofeng
# @File    : train.py
# @Software: PyCharm

import torch
import os
from utils import make_project_dir,save_image
from loss_utils import LossWriter
from UNet import UNet
import torch.optim as optim
import torch.nn as nn
from datasets import SegDatasets


# 定义分割网络的训练和验证超参数
BETA1 = 0.9
BETA2 = 0.999
DATA_ROOT = "human_dataset"
INPUT_DIR_NAME = "human"
LABEL_DIR_NAME = "mask"
LR = 0.0001
BATCH_SIZE = 8
H_FLIP = True
V_FLIP = True
RESULTS_DIR = "results"
EPOCHS = 50
IMAGE_SIZE = 224
IMG_SAVE_FREQ = 100
PTH_SAVE_FREQ = 2

VAL_BATCH_SIZE = 1
VAL_FREQ = 1

device = torch.device("cuda")

# 构建训练和验证dataloader
train_dataset = SegDatasets(IMAGE_SIZE, DATA_ROOT, INPUT_DIR_NAME, LABEL_DIR_NAME, H_FLIP, V_FLIP, train=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
val_dataset = SegDatasets(IMAGE_SIZE, DATA_ROOT, INPUT_DIR_NAME, LABEL_DIR_NAME, H_FLIP, V_FLIP, train=False)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=VAL_BATCH_SIZE,
                                             shuffle=True)


# 定义BCE损失
bce_func = nn.BCELoss()

# 定义分割网络，并将网络参数绑定到Adam优化器
seg_net = UNet().to(device)
optimizer = optim.Adam(params=seg_net.parameters(),
                       lr=LR,
                       betas=(BETA1, BETA2))

make_project_dir(RESULTS_DIR, RESULTS_DIR)
loss_writer = LossWriter(os.path.join(RESULTS_DIR, "loss"))


def train():
    iteration = 0
    for epo in range(1, EPOCHS):
        # 遍历dataloader中所有的数据
        for data in train_loader:
            human = data["human"].to(device)
            mask = data["mask"].to(device)

            # 将人像图片human输入到分割网络中，获得预测的掩码mask
            predict_mask = seg_net(human)

            # 通过预测mask和真实mask计算损失值
            bce_loss = bce_func(predict_mask, mask)

            # 清空梯度，更新网络参数
            optimizer.zero_grad()
            bce_loss.backward()
            optimizer.step()

            # 记录分割损失值，并输出到控制台
            loss_writer.add("bce_loss", bce_loss.item(), iteration)

            print("Iter: {}, BCE Loss: {:.4f}".format(iteration,
                                              bce_loss.item()))
            # 更新迭代次数
            iteration += 1

            # 保存训练过程中的分割mask
            if iteration % IMG_SAVE_FREQ == 0:
                train_patch = torch.cat((predict_mask, mask), dim=3)
                save_image(train_patch[0],
                           out_name=os.path.join(RESULTS_DIR, "train_images",
                                                str(iteration) + ".png"))
        # 保存分割网络权重
        if epo % PTH_SAVE_FREQ == 0:
            torch.save(seg_net.state_dict(),
                       os.path.join(RESULTS_DIR, "pth", str(epo) + ".pth"),
                       _use_new_zipfile_serialization=False)

        # 计算验证集的分割效果
        if epo % VAL_FREQ == 0:
            seg_net.eval()
            with torch.no_grad():
                # 遍历验证集，并进行分割
                for data in val_loader:
                    human = data["human"].to(device)
                    mask = data["mask"].to(device)
                    img_name = data["img_name"]
                    predict_mask = seg_net(human)
                    val_patch = torch.cat((predict_mask, mask), dim=3)
                    # 保存验证集分割结果，用于观察模型表现
                    save_image(val_patch[0],
                               out_name=os.path.join(RESULTS_DIR, "val_images",
                                                      img_name[0]))
            seg_net.train()


if __name__ == "__main__":
    train()
