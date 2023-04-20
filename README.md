# ImageNet-Resnet-start-from-the-scratch-
1.ImageNet数据的下载和处理  
2.利用pytorch读取数据并生成DataLoader  
3.搭建ResNet网络并在该数据集上训练  

## Catalogue
- [x] ImageNet数据集的下载
- [x] ImageNet数据集的处理
- [X] ImageNet数据集的读入
- [ ] ResNet网络的搭建
- [ ] 在CIFAR100数据集上训练ResNet网络

## DataSets
#### ImageNet数据集
**[ImageNet数据集的预处理操作](https://github.com/Evanwu1125/ImageNet-Resnet-start-from-the-scratch-/tree/main/Datasets)**  
ImageNet数据集下载链接  
链接：https://pan.baidu.com/s/1i6arUzJdUH_f3aY2cCJsWA?pwd=evan  
提取码：evan  
提取完数据集之后，验证集是一个包含了50000张图片的文件夹（一共1000个种类，每个种类有50张图片）。  
训练集处理起来稍微有一点麻烦，在解压完训练集后，会发现训练集中还包含了1000个压缩包，其中每个压缩包中还有约为1300张图片。  
我们希望对训练集和验证集处理完之后，文件的目录格式如下图所示：  
```tree
├─train
│  ├─n01440764 #训练集中的每一个文件中有1300张图片
│  ├─n01443537
│  └─n01484850	
└─val
    ├─n01440764 #验证集的每一个文件中只有500张图片
    ├─n01443537
    └─n01484850
```
#### Flower数据集
因为ImageNet数据集用一般的电脑训练起来实在是太困难了，所以我们在检验模型的时候会用这个Flower数据集作为平替。
## Networks
- [ ] Resnet
- [ ] [Vggnet](https://github.com/Evanwu1125/ImageNet-Resnet-start-from-the-scratch-/tree/main/models/VggNet)
- [ ] Googlenet
- [ ] SEnet
- [ ] ResNeXt

（仍在更新中）
