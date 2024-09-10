import os
import random
def list_subdirectories(directory):
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print("提供的路径不是一个有效的目录")
        return

    # 获取目录下的所有项
    items = os.listdir(directory)
    print('子项数量为:',len(items))

    # 过滤出所有子目录
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    print('子目录总数为：',len(subdirectories))

    # 检查所有子目录
    for subdir in subdirectories:
        path=directory+'/'+subdir
        items0 = os.listdir(path)
        # print(len(items0),subdir)
        if not len(items0) == 3:
            print(len(items0),subdir)

    # 进行训练集和验证集的划分
    outData=''
    i=0
    j=0
    k=0
    for subdir in subdirectories:
        name=subdir.split('_')[1]
        random_number = random.randint(1, 40)
        if random_number == 1:#归入验证集
            outData=outData+name+',val'+'\n'
            i=i+1
        elif random_number == 2:#归入测试集
            outData=outData+name+',test'+'\n'
            j=j+1
        else:
            outData=outData+name+',train'+'\n'
            k=k+1
    print('验证集大小：',i)
    print('测试集大小：',j)
    print('训练集大小：',k)
    file_path = '../config/bedroom_threed_front_splits_lzc.csv'
    with open(file_path, 'w') as file:
        # 将字符串写入文件
        file.write(outData)

if __name__ == "__main__":
    print('这个脚本的作用为检测数据集。')
    directory_path = '../../Dataset/out-preprocess' # '/path/to/your/directory'
    # 获取命令行参数列表
    import sys
    args = sys.argv
    if len(args) > 1:
        directory_path = args[1]
    list_subdirectories(directory_path)
