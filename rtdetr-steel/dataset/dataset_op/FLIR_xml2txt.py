import xml.etree.ElementTree as ET
import os

# 类别名称到类别编号的映射
class_mapping = {
    "person": 0,
    "bike": 1,
    "car": 2,
    "motor": 3,
    "bus": 4,
    "train": 5,
    "truck": 6,
    "light": 7,
    "hydrant": 8,
    "sign": 9,
    "dog": 10,
    "deer": 11,
    "skateboard": 12,
    "stroller": 13,
    "scooter": 14,
    "other vehicle": 15
}

def convert_xml_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_annotations = []

    for obj in root.iter('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            continue
        class_id = class_mapping[class_name]

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 计算中心点和宽高的相对值
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    # 获取文件名，不包括扩展名
    file_name = os.path.splitext(os.path.basename(xml_file))[0]

    # 保存到YOLO格式的txt文件中
    with open(os.path.join(output_dir, f"{file_name}.txt"), 'w') as f:
        f.write("\n".join(yolo_annotations))

def process_directory(xml_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            convert_xml_to_yolo(os.path.join(xml_dir, xml_file), output_dir)

# 使用示例
xml_dir = r"/data/lihan/FLIR_ADAS_v2/images_thermal_train/images_thermal_train_xml/data"
output_dir = r"/data/lihan/FLIR_ADAS_v2/label_txt/images_thermal_train"
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
process_directory(xml_dir, output_dir)

xml_dir = r"/data/lihan/FLIR_ADAS_v2/images_rgb_train/images_rgb_train_xml/data"
output_dir = r"/data/lihan/FLIR_ADAS_v2/label_txt/images_rgb_train"
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
process_directory(xml_dir, output_dir)

xml_dir = r"/data/lihan/FLIR_ADAS_v2/images_thermal_val/images_thermal_val_xml/data"
output_dir = r"/data/lihan/FLIR_ADAS_v2/label_txt/images_thermal_val"
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
process_directory(xml_dir, output_dir)

xml_dir = r"/data/lihan/FLIR_ADAS_v2/images_rgb_val/images_rgb_val_xml/data"
output_dir = r"/data/lihan/FLIR_ADAS_v2/label_txt/images_rgb_val"
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
process_directory(xml_dir, output_dir)
