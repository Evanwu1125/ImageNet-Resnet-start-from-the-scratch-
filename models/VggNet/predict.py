import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Vgg

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 通过transforms.Compose构建图片预处理步骤
    data_transforms = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

    #加载图片
    img_path = '../img_pth.jpg'
    #判断图片路径是否正确
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    #用PIL中的Image打开图片
    img = Image.open(img_path)
    plt.imshow(img)
    #对图片进行预处理
    img = data_transforms(img)
    #因为我们在预测的时候是单张图片，但是在传入模型中的时候我们需要一个批量的参数，所以我们把图片的批量上的参数扩充为1
    img = torch.unsqueeze(img, dim = 0)

    #读取标签相关的数据
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    #创建模型
    model = Vgg(model_name = "vgg16", num_classes = 5).to(device)
    #加载模型的参数
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist"
    #用load_state_dict可以加载已经训练好的模型的参数
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        #预测该图片的类别
        #这里最后用.cpu()是因为softmax函数不可以处理在gpu的数据
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output,dim = 0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
