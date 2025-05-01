import os
from sklearn.model_selection import train_test_split

# ✅ 修改为你生成的 .h5 文件路径
data_path = "/content/drive/MyDrive/453GroupProject/brats-unet/processed_dataset"
all_ids = [f[:-len("_mri_norm2.h5")] for f in os.listdir(data_path) if f.endswith(".h5")]

# ✅ 划分 train / val / test（8:1:1）
train_ids, val_test_ids = train_test_split(all_ids, test_size=0.2, random_state=21)
val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=21)

print("Using {} images for training, {} for validation, {} for testing.".format(len(train_ids), len(val_ids), len(test_ids)))

train_ids.sort()
val_ids.sort()
test_ids.sort()

# ✅ 修改输出路径，写入 txt 文件到 data_process 文件夹中
save_dir = "/content/drive/MyDrive/453GroupProject/brats-unet/data_process"

with open(os.path.join(save_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_ids))

with open(os.path.join(save_dir, 'valid.txt'), 'w') as f:
    f.write('\n'.join(val_ids))

with open(os.path.join(save_dir, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_ids))
