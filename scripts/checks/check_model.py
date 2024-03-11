import timm
import torch

use_img_size_config = True

model_name = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"

device = 'cuda'
img_size = 640
if use_img_size_config:
    model = timm.create_model(model_name, pretrained=True, img_size=(img_size, img_size), num_classes=256)
    model = model.to(device)
    data_config = timm.data.resolve_model_data_config(model)
    print(data_config)
else:
    model = timm.create_model(model_name, pretrained=True, img_size=None)

x = torch.randn(2, 3, img_size, img_size).to(device)
o = model(x)
print(x.shape)
print(o.shape)
print(model.global_pool)
