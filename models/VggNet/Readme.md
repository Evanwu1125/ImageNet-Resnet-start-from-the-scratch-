
## Vgg网络模型
文章链接 [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)  
文章主要思想：  
**堆叠两个3×3的卷积核可以用来替代5×5的卷积核，堆叠三个3×3的卷积核可以用来替代7×7的卷积核，这样的一种方式可以有效的降低模型参数,用公式表示如下：**
$7 × 7 × C × C = 49C^{2} => 3 × 3 × C × C + 3 × 3 × C × C + 3 × 3 × C × C = 27C^{2}$
## model.py  
该文件夹中存放的是
$Vgg$
模型的搭建代码，其中根据原论文分别搭建
$Vgg11,Vgg13,Vgg16,Vgg19$，
其中后面的数字是由卷积层数 + 3层线性层数构成的，比如
$Vgg11$
，该模型中就是包含了8个卷积层加上3个线性层。
模型示意图：![image](https://user-images.githubusercontent.com/88299572/232652530-64e9a36a-6e89-43e0-a408-8f52e7291062.png)

## main.py
该文件中存放的是程序的主文件，用来训练模型。在训练过程中**model.py**文件会自动调用**model.py**文件当中的
$Vgg$
模型。

## predict.py
该文件中是用来对单张的本地图片进行预测，具体预测方式是更改第21行的代码  


```python
    # load image
    img_path = "../tulip.jpg" #把行代码是本地图片的地址，直接把这行地址改成自己图片的地址即可
```
