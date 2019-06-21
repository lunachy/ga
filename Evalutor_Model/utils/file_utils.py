"""
文件操作工具类  方便操作文件
"""
import os
import os.path


def copy_file(source_path: str, target_path: str):
    """
    复制文件
    :param source_path:
    :param target_path:
    :return:
    """
    write = open(target_path, "w", encoding="utf-8")
    reader = open(source_path, "r", encoding="utf-8")
    write.write(reader.read())
    reader.close()
    write.close()


def get_file_paths(root_path: str, prefix: str = None, postfix: str = None):
    """
    获取一个根目录下所有的文件目录
    :param root_path:  根目录
    :param prefix:  前缀
    :param postfix:  后缀
    :return:
    """
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        root = root.replace("\\", "/")  # 兼容linux格式
        for file in files:
            file_name = file.split("/")[-1]
            flag = True
            if prefix is not None:
                if not file_name.startswith(prefix):
                    flag = False
            if postfix is not None:
                if not file_name.endswith(postfix):
                    flag = False
            if flag:
                file_paths.append("{0}/{1}".format(root, file))
    return file_paths


def generate_summary(path: str, save_path: str):
    """
    生成数据
    :param path  数据路径
    :param save_file  数据存储路径
    :return:
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    path = os.path.abspath(path)  # 获取绝对目录
    file_paths = get_file_paths(root_path=path, postfix=".json")
    top0_file_paths = list(filter(lambda path: "top" not in path, file_paths))
    top1_file_paths = list(filter(lambda path: "-top1-" in path, file_paths))
    top2_file_paths = list(filter(lambda path: "-top2-" in path, file_paths))
    top3_file_paths = list(filter(lambda path: "-top3-" in path, file_paths))
    # 按照顺序排序
    top0_file_paths = sorted(top0_file_paths)
    top1_file_paths = sorted(top1_file_paths)
    top2_file_paths = sorted(top2_file_paths)
    top3_file_paths = sorted(top3_file_paths)

    write0 = open("{0}/data.txt".format(save_path), "w", encoding="utf-8")
    for file in top0_file_paths:
        write0.write("{0}\n".format(file))
    write0.close()

    write1 = open("{0}/data_top1.txt".format(save_path), "w", encoding="utf-8")
    for file in top1_file_paths:
        write1.write("{0}\n".format(file))
    write1.close()

    write2 = open("{0}/data_top2.txt".format(save_path), "w", encoding="utf-8")
    for file in top2_file_paths:
        write2.write("{0}\n".format(file))
    write2.close()

    write3 = open("{0}/data_top3.txt".format(save_path), "w", encoding="utf-8")
    for file in top3_file_paths:
        write3.write("{0}\n".format(file))
    write3.close()


def generate_train_data(path: str):
    """
    name%9!=0 全部都是训练集
    :param path 数据存储路径
    :return:
    """
    file = open(path, "r", encoding="utf-8")
    path = os.path.abspath(path)  # 获取绝对目录
    path = path.replace("\\", "/")
    # print("--debug: path", path)
    file_name = path.split("/")[-1]
    prefix_name = file_name.split(".")[0]

    train_file_path = "{0}/{1}_train.txt".format("/".join(path.split("/")[: -1]), prefix_name)
    # print("--debug:train_file_path", train_file_path)
    write_file = open(train_file_path, "w", encoding="utf-8")
    lines = file.readlines()
    for line in lines:
        name = line.split("/")[-1].split("-")[0]
        if int(name) % 15 != 0:
            write_file.write(line)
    write_file.close()


def generate_test_data(path: str):
    """
    name%9==0 全部都是测试集
    :param path 数据存储路径
    :return:
    """
    file = open(path, "r", encoding="utf-8")
    path = os.path.abspath(path)  # 获取绝对目录
    path = path.replace("\\", "/")
    file_name = path.split("/")[-1]
    prefix_name = file_name.split(".")[0]
    test_file_path = "{0}/{1}_test.txt".format("/".join(path.split("/")[: -1]), prefix_name)
    write_file = open(test_file_path, "w", encoding="utf-8")
    lines = file.readlines()
    for line in lines:
        name = line.split("/")[-1].split("-")[0]
        if int(name) % 15 == 0:
            write_file.write(line)
    write_file.close()
