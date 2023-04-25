import os
import json
import sys

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import Vgg

data_transform = {
    'train':transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
    'val':transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
}

def main():
    #判断本地是否有GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ''''找到数据集所在路径'''
    #这行代码的用途是找到当前文件的向上两级目录的绝对路径
    data_root = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    #把根目录的路径和相关路径结合，找到数据集所在路径
    image_path = os.path.join(data_root, "每周一篇深度学习",'dataset','flower_data')
    assert os.path.exists(image_path), "{} path does not exist".format(image_path)
    #这里的ImageFolder函数可以直接把图片的文件夹打包成训练集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,'train'),
                                         transform=data_transform['train'])

    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunfolwer':3, 'tuplis':4}
    flower_list = train_dataset.class_to_idx
    #制作标签
    cla_dict = dict((val,key) for key, val in flower_list.items())
    #write dict into json files
    json_str = json.dumps(cla_dict,indent=4)
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    #这里的nw是num_workers的缩写
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f"Using {nw} workers every process.")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = nw)


    validate_dataset = datasets.ImageFolder(root = os.path.join(image_path,'train'),
                                            transform=data_transform['val'])
    val_num = len(validate_dataset)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size = batch_size,
                                                  shuffle = True,
                                                  num_workers = nw)


    model_name = 'vgg16'
    net = Vgg(model_name, num_classes = 5, init_weights = True)
    net.to(device)
    #损失函数规定为交叉熵损失
    loss_function = nn.CrossEntropyLoss()
    #优化器用Adam
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    #一共训练30个回合
    epochs = 30
    #最高的准确率
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        #开始训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        #每个回合都验证一次
        net.eval()
        acc = 0.0 #计算每个回合的精确度
        #在每次验证之前把梯度清零
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file = sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim = 1)[1]
                #torch.eq()函数是来判断两个列向量有多少是相同的
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        #算出验证集的准确率
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Finished training!")


if __name__ == '__main__':
    main()
