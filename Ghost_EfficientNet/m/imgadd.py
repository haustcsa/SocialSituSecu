import os
from PIL import Image
import random

# 源文件夹路径
root_path = r'D:\MCY\MMMMM\数据备份\q'

# ['politics', 'porn', 'safe', 'terroristic']
# 指定要增加数据的文件夹
dirs_path = './dataset/safe'

# 是否删除源文件
del_root = False

# 增添数量，超过将会将所有添加
img_num = min(len(os.listdir(root_path)), 4000)

if __name__ == "__main__":
    num = len(os.listdir(dirs_path))
    print(num)  # 增添开始的序号
    img_list = [root_path + "/" + img for img in os.listdir(root_path)]
    random.shuffle(img_list)
    for img in img_list[0:img_num]:
        try:
            print(img, "完成")
            image = Image.open(img).convert("RGB")
            image.save(dirs_path + '/' + str(num) + ".jpg")

            if del_root:
                os.remove(img)
            num = num + 1
        except Exception as e:
            print(img, "失败       错误信息：", e)
