import os
import shutil

# 源文件夹和目标文件夹
src_dir = '/data/lihan/FLIR_ADAS_v1_yolo/thermal/images/train'
dst_dir = '/home/lihan/Code/rtdetr/heatmap/images'

# 需要复制的文件列表
# default_image_list = [
#     'FLIR_00401.jpeg', 'FLIR_00403.jpeg', 'FLIR_00406.jpeg', 'FLIR_00414.jpeg', 'FLIR_00429.jpeg',
#     'FLIR_00447.jpeg', 'FLIR_00448.jpeg', 'FLIR_00493.jpeg', 'FLIR_00575.jpeg', 'FLIR_00581.jpeg',
#     'FLIR_06664.jpeg', 'FLIR_00859.jpeg', 'FLIR_02590.jpeg', 'FLIR_02110.jpeg', 'FLIR_08431.jpeg'
# ]
default_image_list = [
    'FLIR_00549.jpeg', '.jpeg','.jpeg','.jpeg','.jpeg','.jpeg',
    '.jpeg','.jpeg','.jpeg','.jpeg','.jpeg',
]

def copy_files1():
    # 确保目标文件夹存在
    os.makedirs(dst_dir, exist_ok=True)
    
    # 复制文件
    for filename in default_image_list:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Successfully copied {filename}")
        except FileNotFoundError:
            print(f"Error: {filename} not found in source directory")
        except Exception as e:
            print(f"Error copying {filename}: {str(e)}")
            
    print("\nFile copying completed!")
    print(f"Files copied to: {dst_dir}")

def clear_and_copy(filelist):
    # 清除目标文件夹内容
    for filename in os.listdir(dst_dir):
        file_path = os.path.join(dst_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed {filename}")
        except Exception as e:
            print(f"Error removing {filename}: {str(e)}")
    
    # 复制文件
    for filename in filelist:
        
        dst_path = os.path.join(dst_dir, os.path.basename(filename))
        
        try:
            shutil.copy2(filename, dst_path)
            print(f"Successfully copied {filename}")
        except FileNotFoundError:
            print(f"Error: {filename} not found in source directory")
        except Exception as e:
            print(f"Error copying {filename}: {str(e)}")
            
    print("\nFile copying completed!")
    print(f"Files copied to: {dst_dir}")
if __name__ == "__main__":

    # 数据集的图片文件夹路径
    dataset_path = '/data/lihan/FLIR_ADAS_v1_yolo/thermal/images/train'  

    # 获取数据集中的前100张图片路径
    image_paths = sorted(os.listdir(dataset_path))[500:600]
    # 使用挑选出来的路径
    # image_paths = default_image_list

    # 生成全部的完整的image_paths
    image_paths = [os.path.join(dataset_path, img) for img in image_paths]

    clear_and_copy(image_paths)