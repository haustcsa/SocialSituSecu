
import os
import shutil

def recursive_listdir(path):
    files = os.listdir(path)
    num = 0
    newdir = "G:\DFDC\dfdc_train_part_00\dataset"
    newdir1 = '\\'+'test_set'
    final_dir = newdir + newdir1
    for file in files:
        file_path = os.path.join(path, file)
        num = num + 1
        if os.path.isdir(file_path) and num < 196:
            print(file)
            src = os.path.join(path,file)
            dst = os.path.join(final_dir, file)
            print('src:', src)
            print('dst:', dst)
            shutil.move(src, dst)
        else:
            print(num)
            break


# recursive_listdir(r'./test')
recursive_listdir('G:\DFDC\dfdc_train_part_00\output')
