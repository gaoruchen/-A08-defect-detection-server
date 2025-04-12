import os
import shutil

def copy_folder_contents(src_folder, dst_folder):
    """
    将源文件夹中的所有内容复制到目标文件夹中。

    参数:
        src_folder (str): 源文件夹路径。
        dst_folder (str): 目标文件夹路径。
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 遍历源文件夹中的所有内容
    for item in os.listdir(src_folder):
        # 构建源路径和目标路径
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        # 如果是文件，则复制文件
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"复制文件: {src_path} -> {dst_path}")
        # 如果是文件夹，则递归复制文件夹
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
            print(f"复制文件夹: {src_path} -> {dst_path}")
def add_suffix_to_jpeg_files(folder_path, suffix):
    """
    给文件夹内所有 .jpeg 文件加上后缀。

    参数:
        folder_path (str): 文件夹路径。
        suffix (str): 要添加的后缀（例如 "_new"）。
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是 .jpeg 文件
        if filename.lower().endswith('.jpeg'):
            # 获取文件的完整路径
            file_path = os.path.join(folder_path, filename)
            # 分离文件名和扩展名
            name, ext = os.path.splitext(filename)
            # 构建新的文件名
            new_filename = f"{name}{suffix}{ext}"
            # 构建新的文件完整路径
            new_file_path = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"重命名: {filename} -> {new_filename}")

if __name__ == '__main__':

    

    # add_suffix_to_jpeg_files("heatmap_result_r18", "ar18")
    # add_suffix_to_jpeg_files("heatmap_result_LARM", "LARM")
    # add_suffix_to_jpeg_files("heatmap_result_ASFM", "ASFM")
    # add_suffix_to_jpeg_files("heatmap_result_LCAF", "LCAF")
    # add_suffix_to_jpeg_files("heatmap_result_LCAF_USE", "LCAF_USE")


    # 拷贝内容到 heatmap_result
    destination_folder = "heatmap_result"
    if os.path.exists(destination_folder):
        # 删除目标文件夹及其内容
        shutil.rmtree(destination_folder)
        print(f"清空文件夹: {destination_folder}")
    # 重新创建空的目标文件夹
    os.makedirs(destination_folder)
    print(f"创建文件夹: {destination_folder}")
    folders_to_copy = [
        "heatmap_result_r18",
        # "heatmap_result_LARM",
        "heatmap_result_ASFM",
        # "heatmap_result_LCAF",
        # "heatmap_result_LCAF_USE"
    ]
    

    for folder in folders_to_copy:
        copy_folder_contents(folder, destination_folder)
