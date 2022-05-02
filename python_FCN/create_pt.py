import torch
# from UNet import UNet
from fcn import FCN

device = torch.device("cpu")
seg_net = FCN(num_classes=1)
seg_net.load_state_dict(torch.load("results_FCN/pth/20.pth", map_location=device))
seg_net.eval()
x = torch.rand(1, 3, 256, 256)
traced_script_module = torch.jit.trace(func=seg_net, example_inputs=x)
traced_script_module.save("20_fcn.pt")
