import torch
import torch.nn as nn
import os
import json
import sys

from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import ..

#数据的预处理操作
data_transforms = {
    'train':transforms.Compose([
        # 因为大多数模型的输入图片尺寸都是224
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    'val':transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}

def main():
    #判断本地是否有gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
    batch_size = 32
    #判断电脑一共可以有多少线程同时工作
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8 ])
    #训练集和验证集所在的文件夹
    train_image_path = '../..'
    test_image_path = '../..'
    assert train_image_path, "{} does not exist".format(train_image_path)
    assert test_image_path,"{} does not exist".format(test_image_path)
    # ImageFolder函数可以直接把图片的文件夹打包成训练集
    train_dataset = datasets.ImageFolder(root = train_image_path, transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(root = test_image_path, transform=data_transforms['val'])
    # 构造train和test数据集
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              num_workers = num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size = batch_size,
                                              shuffle = False,
                                              num_workers = num_workers)
    # 标签制作
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # indent这个参数表示json文件中每行的缩进程度
    json_str =json.dumps(cla_dict, indent=4)
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)


    train_num = len(train_loader); test_num = len(test_loader)
    model_name = '...'
    net = '...'
    net.to(device)
    #损失函数定义为交叉熵损失
    loss_function = nn.CrossEntropyLoss()
    #优化器用Adam
    optimizer = optim.Adam(net.paramaters(), lr = 0.0001)

    #一共训练30个回合
    epochs = 30
    #最高的准确率
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
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
            val_bar = tqdm(test_loader, file = sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim = 1)[1]
                #torch.eq()函数是来判断两个列向量有多少是相同的
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        #算出验证集的准确率
        val_accurate = acc / test_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_num, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Finished training!")




