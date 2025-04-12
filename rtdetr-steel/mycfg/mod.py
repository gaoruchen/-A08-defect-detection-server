import os

def rename_yaml_files(path, old_name, new_name):
    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(path):
        # 检查文件是否以 .yaml 结尾
        if filename.endswith('.yaml'):
            # 检查文件名中是否包含 old_name
            if old_name in filename:
                # 替换 old_name 为 new_name
                new_filename = filename.replace(old_name, new_name)
                # 获取文件的完整路径
                old_file_path = os.path.join(path, filename)
                new_file_path = os.path.join(path, new_filename)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {filename} -> {new_filename}')

def list_yaml_files(path, memo="memo"):
    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(path):
        # 检查文件是否以 .yaml 结尾
        if filename.endswith('.yaml'):
            # 去除 .yaml 后缀
            name_without_extension = filename[:-5]  # 去掉最后 5 个字符（.yaml）
            # 打印文件名和 memo
            print(f'"{name_without_extension}:{memo}"')




if __name__ == '__main__':

    # rename_yaml_files('/home/lihan/Code/rtdetr/mycfg/original', 'AIFIRepBN', 'AIFI_SLA')

    list_yaml_files('/home/lihan/Code/rtdetr/mycfg', memo="memo")