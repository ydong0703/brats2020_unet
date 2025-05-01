import h5py
import numpy as np

# ✅ 修改路径为 Google Drive 上的 BraTS2020 文件
p = '/content/drive/MyDrive/453GroupProject/brats-unet/processed_dataset/BraTS20_Training_001_mri_norm2.h5'

# 读取数据
h5f = h5py.File(p, 'r')
image = h5f['image'][:]
label = h5f['label'][:]

print('image shape:', image.shape, '\t', 'label shape:', label.shape)
print('label set:', np.unique(label))
