# Classification-cifar10-pytorch  
I am testing several classical classification networks performance on cifar10 dataset by PyTorch!  
# Requirements  
- pytorch  
- torchsummary  
- python3.x  
# Results  
| Model             | My Acc.        | Total params  |  Estimated Total Size (MB) |  Trainable params | Params size (MB) |Saved model size (MB)|GPU memory usage(MB)
| ----------------- | ----------- | ------  | ---|--- | --- |  --- |--- |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 92.64%      | 2,296,922 | 36.14 | 2,296,922 | 8.76 | 8.96 | 3107 |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 94.27%      | 14,728,266  | 62.77  |14,728,266 |56.18 |59.0 | 1229 |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 94.70%      | 11,171,146 | 53.38  | 11,171,146 | 42.61  | 44.7  |  1665 |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 95.09%      | 9,128,778 | 99.84 | 9,128,778 | 34.82  | 36.7  |  5779 |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 95.22%      | 23,520,842 | 155.86 | 23,520,842  | 89.72 | 94.4 | 5723 |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.42%      | 34,236,634 | 243.50 | 34,236,634  | 130.60 | 137.5  |  10535 |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 95.49%      | 4,774,218 | 83.22 | 4,774,218 | 18.21 | 19.2  |  5817 |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.55%      | 6,956,298 | 105.05 | 6,956,298 | 26.54  |  28.3 |  8203 |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 95.59%      | 11,173,962 | 53.89 | 11,173,962  | 42.63 | 44.8 | 1615 |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 95.62%      | 42,512,970 | 262.31  | 42,512,970  | 162.17 | 170.6  |  8857 |  
**Note**:   
1. Above GPU memory usage(MB) was observed with batch size=128.     
2. For PreActResNet18, I set initial learning rate=0.1, but it can't converge, so I set it's initial lr=0.01.    
3. When I firstly train **VGG16**, **ResNet18** and **ResNet50** with total epochs=400. But I want to get results earlier, so for remaining networks, I set total epochs=300 (besides, afterwards it just improve a little).
# Pre-trained models  
You can obtain pre-traind models(as above list) from here:
[[Baidu Drive](https://pan.baidu.com/s/1oUfaxFnghIdClCFMf3A11Q)] [[Google Drive](https://drive.google.com/open?id=1PLwxkczvKq86ATRD7SB-5w31omuORUNV)]

