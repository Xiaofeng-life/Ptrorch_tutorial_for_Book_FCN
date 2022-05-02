import torch
# from UNet import UNet
from old_unet import UNet
# from fcn import FCN
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    device = torch.device("cpu")
    seg_net = UNet().to(device)
    # seg_net = FCN(num_classes=1).to(device)
    # seg_net.load_state_dict(torch.load("results_unet/pth/30.pth"))
    # torch.save(seg_net.state_dict(),
    #            "results/pth/30.pth",
    #            _use_new_zipfile_serialization=False)

    seg_net.load_state_dict(torch.load("results/pth/30.pth",
                                       map_location="cpu"))
    seg_net.eval()
    image_path = "human_dataset/val_human/00031.png"

    with torch.no_grad():
        human_image = Image.open(image_path)
        human_image = human_image.resize((256, 256))
        human_image = TF.to_tensor(human_image).to(device).unsqueeze(0)
        # 获取分割掩码
        predict_mask = seg_net(human_image)
        # 以0.5作为阈值区分前景和背景
        predict_mask[predict_mask > 0.5] = 1
        predict_mask[predict_mask <= 0.5] = 0

        human_image = human_image * 255
        predict_mask = predict_mask * 255
        # 将预测掩码复制为3通道
        predict_mask = torch.cat((predict_mask, predict_mask, predict_mask), dim=1)

        # 拼接人像和掩码
        result = torch.cat((human_image, predict_mask), dim=3)[0]
        result = result.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)

        plt.imshow(result)
        plt.savefig("pictures/demo.png",  dpi=500, bbox_inches='tight')
        plt.show()
