import h5py
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
# 四种模态的mri图像
modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
    'root': '/content/drive/MyDrive/453GroupProject/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
    'out': '/content/drive/MyDrive/453GroupProject/brats-unet/processed_dataset',
    'flist': '/content/drive/MyDrive/453GroupProject/brats-unet/data_process/train.txt',
}




def process_h5(path, out_path):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    # SimpleITK读取图像默认是是 DxHxD，这里转为 HxWxD
    label = sitk.GetArrayFromImage(sitk.ReadImage(path + 'seg.nii.gz')).transpose(1,2,0)
    print(label.shape)
    # 堆叠四种模态的图像，4 x (H,W,D) -> (4,H,W,D) 
    images = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(path + modal + '.nii.gz')).transpose(1,2,0) for modal in modalities], 0)  # [240,240,155]
    # 数据类型转换
    label = label.astype(np.uint8)
    images = images.astype(np.float32)
    case_name = path.split('/')[-1]
    # case_name = os.path.split(path)[-1]  # windows路径与linux不同
    
    path = os.path.join(out_path,case_name)
    output = path + 'mri_norm2.h5'
    # 对第一个通道求和，如果四个模态都为0，则标记为背景(False)
    mask = images.sum(0) > 0
    for k in range(4):

        x = images[k,...]  #
        y = x[mask]

        # 对背景外的区域进行归一化
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[k,...] = x
    print(case_name,images.shape,label.shape)
    f = h5py.File(output, 'w')
    f.create_dataset('image', data=images, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()


def doit(dset):
    root, out_path = dset['root'], dset['out']
    file_list = dset['flist']

    with open(file_list, 'r') as f:
        subjects = f.read().splitlines()

    # 构建每个样本的路径前缀：/.../BraTS20_Training_001/BraTS20_Training_001_
    paths = [os.path.join(root, case, case + '_') for case in subjects]

    for path in tqdm(paths):
        process_h5(path, out_path)

    print('Finished')



if __name__ == '__main__':
    doit(train_set)
