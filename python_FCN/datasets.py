from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from augument import horizontal_flip, vertical_flip
import numpy as np


class SegDatasets(Dataset):
    def __init__(self, image_size, data_root, input_dir_name,
                 label_dir_name, h_flip, v_flip, train=True):
        # 定义属性
        self.image_size = image_size
        self.data_dir = data_root
        self.input_dir_name = input_dir_name
        self.label_dir_name = label_dir_name
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.train = train
        if self.train:
            self.prefix = "train_"
        else:
            self.prefix = "val_"

        # 检查目录是否存在
        if not os.path.exists(self.data_dir):
            raise Exception(r"[!] data set does not exist!")

        # 获取所有训练数据的名称，存储到self.files列表中
        self.files = sorted(os.listdir(os.path.join(self.data_dir,
                                                    self.prefix + self.input_dir_name)))

    def __getitem__(self, item):
        file_name = self.files[item]
        # 打开img和对应的mask，file_name[:-4] + "_matte.png"代表原始数据集中mask的命名方式
        img = Image.open(os.path.join(self.data_dir,
                                      self.prefix + self.input_dir_name,
                                      file_name)).convert('RGB')
        mask = Image.open(os.path.join(self.data_dir,
                                       self.prefix + self.label_dir_name,
                                       file_name[:-4] + "_matte.png")).convert('L')

        # 将img和mask进行resize，统一为相同尺寸
        img = TF.resize(img, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size))

        if self.train:
            # 以0.5的概率进行数据增强，增强方式必须保证img和mask的变换是完全对应的
            if self.h_flip and np.random.random() > 0.5:
               img, mask = horizontal_flip(img, mask)

            if self.v_flip and np.random.random() > 0.5:
                img, mask = vertical_flip(img, mask)

        # 将图像转为tensor类型
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        # 以字典形式返回img、mask和img的名字
        out = {'human': img, 'mask': mask, "img_name": file_name}

        return out

    def __len__(self):        return len(self.files)


if __name__ == "__main__":
    BETA1 = 0.9
    BETA2 = 0.999
    DATA_ROOT = "human_dataset"
    INPUT_DIR_NAME = "human"
    LABEL_DIR_NAME = "mask"
    LR = 0.0001
    BATCH_SIZE = 16
    H_FLIP = True
    V_FLIP = True
    RESULTS_DIR = "results"
    EPOCHS = 50
    IMAGE_SIZE = 224
    IMG_SAVE_FREQ = 100
    PTH_SAVE_FREQ = 2

    VAL_BATCH_SIZE = 1
    VAL_FREQ = 1
    # 构建训练Dataset
    train_set = SegDatasets(IMAGE_SIZE, DATA_ROOT, INPUT_DIR_NAME, LABEL_DIR_NAME, H_FLIP, V_FLIP, train=True)

    # 数据集中图像数量
    print("num of Train set {}".format(len(train_set)))

    # 获取数据集中的第3条数据的原始图像img、掩码mask和图像名称
    img = train_set[3]["human"]
    mask = train_set[3]["mask"]
    name = train_set[3]["img_name"]

    # 展示原始图像img和掩码mask
    plt.subplot(1, 2, 1)
    plt.imshow(img.numpy().transpose(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(mask.numpy().squeeze())
    plt.show()
