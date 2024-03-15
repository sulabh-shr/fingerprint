import timm
import torch
import timm.models.swin_transformer


class Temp(torch.nn.Module):
    def __init__(self, _model):
        super().__init__()
        self._model = _model

    def forward(self, _x):
        return self._model(_x)



# model_name = "efficientvit_m5.r224_in1k"
# model_name = "samvit_base_patch16.sa1b"
# model_name = "vit_base_patch16_clip_384.openai_ft_in1k"
# model_name = "vit_base_patch16_384.augreg_in21k_ft_in1k"
# model_name = "vit_base_patch16_384.orig_in21k_ft_in1k"
# model_name = "vit_large_patch16_384.augreg_in21k_ft_in1k"
# model_name = "vit_huge_patch14_224.orig_in21k"
# model_name = "swin_large_patch4_window12_384.ms_in22k"
# model_name = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
# model_name = "efficientvit_m5.r224_in1k"
model_name = "mobilenetv3_large_100.ra_in1k"
use_img_size_config = False

device = 'cuda'
img_size = 224
if use_img_size_config:
    model = timm.create_model(model_name, pretrained=True, img_size=img_size, num_classes=256)
    model = model.to(device)
else:
    model = timm.create_model(model_name, pretrained=True, img_size=None, num_classes=256)
    model.to(device)
data_config = timm.data.resolve_model_data_config(model)
print(data_config)
x = torch.randn(2, 3, img_size, img_size).to(device)


temp = Temp(model)
temp.train()
print(f'Input shape: {x.shape}')
# o = model(x)
# print(f'Output shape: {o.shape}')
f = model.forward_features(x)
print(f'Feature shape: {f.shape}')
f2 = model(x)
print(f'Final feature shape: {f2.shape}')

params = 0
for p in temp.parameters():
    params += p.numel()
print(f'Total parameters: {params/1e6} M')
