import tarfile
import sys
import os
def get_file(path):
    files = os.listdir(path)
    return files

if __name__ == '__main__':
    init_path = "ILSVRC2012_img_train"
    new_path = "train"
    files = get_file(init_path)
    not_exist_files = []
    total_name = []
    for file in files:
        name = file.split('.')[0]
        total_name.append(name)
        if not os.path.exists(new_path + '/' + name):
            os.mkdir(path= new_path + '/' + name)
            print(f"创建{name}文件夹成功")
            not_exist_files.append(file)
        else:
            print("已存在文件夹")

    for file_name in not_exist_files:
        #原来的路径
        init_file_path = init_path + '/' + file_name
        #新的路径
        tar_file_path = new_path + '/' + file_name.split('.')[0]
        tar = tarfile.open(init_file_path)
        tar.extractall(tar_file_path)

