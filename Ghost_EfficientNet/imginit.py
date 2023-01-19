import os
from PIL import Image
import random


def img_init():
    # 文件夹路径
    dirs_path = './dataset'
    # 指定文件夹，若不指定则为所有👇
    # dir_list = ['politics', 'porn', 'safe', 'terroristic']
    dirs_list = ['porn']
    dirs = dirs_list if len(dirs_list) is not 0 else os.listdir(dirs_path)
    all_num = 0
    for dir_name in dirs:
        dir_path = os.path.join(dirs_path, dir_name)
        all_num = all_num + len(os.listdir(dir_path))
        print("类别：{}     数量：{}".format(dir_name, len(os.listdir(dir_path))))
        for i, img_name in enumerate(os.listdir(dir_path)):
            img_path = dir_path + "/" + img_name
            try:
                print(img_path, "完成")
                img = Image.open(img_path).convert("RGB")
                img.save(dir_path + "/" + str(i) + ".jpg")
                os.remove(img_path)
            except Exception as ex:
                print(img_path, "失败")
                os.remove(img_path)
                print(ex)
    print("总数：{}".format(all_num))


def ukraine_war_handler():
    path = r"D:\MCY\MMMMM\数据备份\ukraine_war"
    i = 0
    for img in os.listdir(path):
        if img.endswith(('png', 'jpg')):
            pass
        else:
            name_list = img.split('.')
            new_name = "rename" + str(i) + "." + name_list[1]
            print(new_name)
            i = i + 1
            os.rename(os.path.join(path, img), os.path.join(path, new_name))


def Quad_Leaders_handler():
    path = r"D:\MCY\MMMMM\数据备份\Quad Leaders"
    path1 = path.replace("Quad Leaders", "Quad Leaders_1")
    if not os.path.exists(path1):
        os.mkdir(path1)
    print(path1)
    path2 = path.replace("Quad Leaders", "Quad Leaders_2")
    if not os.path.exists(path2):
        os.mkdir(path2)
    print(path2)

    lists = [leader_name for leader_name in os.listdir(path) if os.path.isdir(path + "/" + leader_name)]
    for each_leader in lists:
        _path1 = path1 + "/" + each_leader
        _path2 = path2 + "/" + each_leader
        _path = path + "/" + each_leader
        if not os.path.exists(_path1):
            os.mkdir(_path1)
        if not os.path.exists(_path2):
            os.mkdir(_path2)
        for img in os.listdir(_path):
            if not img.endswith("_c.jpg"):
                Image.open(_path + "/" + img).save(_path1 + "/" + img)
            else:
                Image.open(_path + "/" + img).save(_path2 + "/" + img)


if __name__ == "__main__":
    img_init()
    # Quad_Leaders_handler()
    # ukraine_war_handler()
