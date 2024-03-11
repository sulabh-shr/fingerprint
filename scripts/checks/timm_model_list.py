import timm.models.vision_transformer
import timm.models.swin_transformer

# filter = '*efficient*'
# filter = '*sam*'
# filter = 'vit_huge*'
# filter = 'vit_base*'
filter = '*swin*'

avail_pretrained_models = timm.list_models(filter, pretrained=True)
for i in avail_pretrained_models:
    print(i)
