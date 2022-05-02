import shutil
import os
from PIL import Image
import numpy as np


def save_image(image_tensor, out_name):
    """
    save a single image
    :param image_tensor: torch tensor with size=(3, h, w)
    :param out_name: path+name+".jpg"
    :return: None
    """
    # print("in size", image_tensor.size())
    if len(image_tensor.size()) == 3:
        # image_numpy = image_tensor.cpu().detach().numpy().transpose(1, 2, 0)
        image_numpy = image_tensor.cpu().detach().numpy().squeeze(0)
        image_numpy[image_numpy > 0.5] = 255
        image_numpy[image_numpy < 0.5] = 0
        # image_numpy = (((image_numpy + 1) / 2) * 255).astype(np.uint8)
        image_numpy = image_numpy.astype(np.uint8)
        image = Image.fromarray(image_numpy)
        image.save(out_name)
    else:
        raise ValueError("input tensor not with size (3, h, w)")
    return None

def check_mk_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def split(ori_dir, tar_dir):
    files = os.listdir(ori_dir)
    for file in files:
        if len(file) > 10:
            shutil.copy(os.path.join(ori_dir, file), os.path.join(tar_dir, file))


def make_sub_dirs(base_dir, sub_dirs):
    """
    six
    :param base_dir:
    :param sub_dirs:
    :return:
    """
    check_mk_dir(base_dir)
    for sub_dir in sub_dirs:
        check_mk_dir(os.path.join(base_dir, sub_dir))


def make_project_dir(train_dir, val_dir):
    make_sub_dirs(train_dir, ["train_images", "pth", "loss"])
    make_sub_dirs(val_dir, ["val_images"])



if __name__ == "__main__":
    pass
    # writer = SummaryWriter('logs')
    #
    # for i in range(100):
    #     writer.add_scalar("test/sin", np.sin(i), i)
    #     writer.add_scalars("test1", {"sin": np.sin(i), "cos": np.cos(i)}, i)
    # # step4：close
    # writer.close()
