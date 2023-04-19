### ImageNet数据集
**ILSVRC2021_img_train**文件存放的是从百度网盘里解压好的
$ImageNet$
训练集  
**ILSVRC2021_img_val**文件存放的是从百度网盘里解压好的
$ImageNet$
验证集  
**Imagenet_train_zip.py**是用来解压
$ImageNet$
训练集文件夹中的1000个子压缩件的，其中每一个压缩的文件夹都会以相同的名字出现在
$train$
文件夹中  
**Imagenet_val_process.py**是用来整理
$ImageNet$
验证集文件夹中的50000张图片的，其中每一个整理好的的文件夹都会以相同的名字出现在
$val$
文件夹中  
**imagenet_val.txt**装的是
$ImageNet$
的验证集标签

### Flower数据集
**flower_photos**该文件夹中装的花的分类数据集。因为
$ImageNet$
数据集跑起来成本太高，所以我们用小的数据集来对我们的网络进行实验。
