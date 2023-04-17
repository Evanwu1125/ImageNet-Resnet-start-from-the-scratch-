
## Vgg网络模型
文章链接 [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)  
文章主要思想：  
**堆叠两个3×3的卷积核可以用来替代5×5的卷积核，堆叠三个3×3的卷积核可以用来替代7×7的卷积核，这样的一种方式可以有效的降低模型参数,用公式表示如下：**
$7 × 7 × C × C = 49C^{2} => 3 × 3 × C × C + 3 × 3 × C × C + 3 × 3 × C × C = 27C^{2}$
## model.py
