## ResNet网络模型
文章链接：https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1512.03385  
**文章的主要思想:**  
关于
$ResNet$
的思想来源，个人觉得这篇[知乎文章](https://zhuanlan.zhihu.com/p/101332297)讲的是最好的，我在这里就简单的总结一下：
其实**残差连接**的思想溯源于一个信念。

因为
$AlexNet$
在2012的**ILSVRC（ImageNet Large Scale Visual Recognition Challenge）中取得了冠军**，再加上后来的
$GoogleNet$
和
$InceptionNet$
等模型都好像证明了网络的层数越深就可以带来更高的准确率，所以大家一时间都认为只要我模型叠得够深就可以获得更高的精度。但何恺明在
$ResNet$
论文中证明了当层数到达一定层数时，继续加深模型不仅不会使模型的准确率上升， 反而会让模型的效果下降，
$ResNet$团队把这样一种现象称为
$Degradation$(退化)。

其实得到这样一个结论是不符合大家对深度学习的认知的，因为大家都觉得深度学习这个黑箱可以模拟任何函数，但是这个函数当维度高到一定程度时竟然不能很好的保留前一层的特征。
这其实是因为在深度学习中有大量的非线性激活函数
