import os
import json
from pycocotools.coco import COCO
coco_to_yolo_id_map = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79
}

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(coco, img_id, output_dir):
    img = coco.loadImgs(img_id)[0]
    width = img['width']
    height = img['height']

    ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    with open(os.path.join(output_dir, '%s.txt' % img['file_name'][:-4]), 'w') as out_file:
        for ann in anns:
            category_id = ann['category_id']
            category_id = coco_to_yolo_id_map[category_id]
            bbox = ann['bbox']
            # COCO format: [x, y, width, height]
            x, y, w, h = convert((width, height), bbox)
            out_file.write(f"{category_id} {x} {y} {w} {h}\n")

def main():
    

    data_dir = '/data/lihan/coco'
    ann_file = os.path.join(data_dir, 'annotations/instances_train2017.json')
    output_dir = '/data/lihan/coco/labels/train'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        convert_annotation(coco, img_id, output_dir)

    
    ann_file = os.path.join(data_dir, 'annotations/instances_val2017.json')
    output_dir = '/data/lihan/coco/labels/val'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        convert_annotation(coco, img_id, output_dir)

if __name__ == "__main__":
    main()
