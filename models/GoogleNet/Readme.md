## GoogleNet网络模型
文章链接[Going deeper with convolutions](https://arxiv.org/pdf/1409.4842v1.pdf)  
文章主要思想:  
1.网络中使用了模块化的结构**Inception模块**以保证在增加层数的时候避免参数量过大。  
2.本文在避免梯度消失和正则化方面提出了两个辅助分类器(**只有在训练的时候会使用这两个辅助分类器，验证的时候不使用**)


![image](https://user-images.githubusercontent.com/88299572/235573076-4b71166b-7899-48b4-964a-e065f9b31733.png)
## model.py
该文件中存放的是
$GoogleNet$
模型的搭建代码。
## predict.py
该文件中是用来对单张的本地图片进行预测，具体预测方式是更改第20行的代码
```python
    # load image
    img_path = "../tulip.jpg" #把行代码是本地图片的地址，直接把这行地址改成自己图片的地址即可
```
##

