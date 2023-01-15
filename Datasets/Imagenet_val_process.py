##生成val数据的每个文件夹
import os
import sys
import shutil #移动文件夹

## 先生成1000个验证集
train_path = 'train'
val_path = 'val'
# files = os.listdir(path=train_path)
# for file in files:
#     file_name = val_path + '/' + file
#     os.mkdir(file_name)

#读入验证集的txt文件
txt_file = open('imagenet_val.txt')
val_root_path = 'ILSVRC2012_img_val'
for line in txt_file:
    name, category = line.split(' ')[1], line.split(' ')[-1][:9]
    init_path = val_root_path + '/' + name
    new_path = val_path + '/' + category

    shutil.move(init_path, new_path)

