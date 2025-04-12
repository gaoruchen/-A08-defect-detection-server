import os
 
train_path = r"/data/lihan/FLIR_ADAS_v2/label_txt/images_thermal_train"
val_path = r"/data/lihan/FLIR_ADAS_v2/label_txt/images_thermal_val"
 
out = r"/home/lihan/Code/rtdetr/dataset/FLIRv2/"
 
train = os.listdir(train_path)
val = os.listdir(val_path)
 
for i in train:
    with open(out+'train_list.txt', 'a+') as f:
        f.write("/data/lihan/FLIR_ADAS_v2/images_thermal_train/data"+ i + '\n')
 
for i in val:
    with open(out+'val_list.txt', 'a+') as f:
        f.write("/data/lihan/FLIR_ADAS_v2/images_thermal_val/data"+ i + '\n')