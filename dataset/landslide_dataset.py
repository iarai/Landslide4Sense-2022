import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import h5py


class LandslideDataSet(data.Dataset):
    def __init__(self, data_dir, list_path, set_mask='masked', transform=None):
        self.mean = [0.9257, 0.9227, 0.9541, 0.9596, 1.0228, 1.0426, 1.0358, 1.0468, 1.1699, 1.1736, 1.0495, 1.0370,
                     1.2511, 1.6495]
        self.std = [0.1410, 0.2207, 0.3184, 0.5724, 0.4601, 0.4465, 0.4651, 0.4948, 0.5133, 0.6836, 0.5323, 0.6628,
                    0.6784, 1.0727]
        self.set_mask = set_mask
        self.transform = transform
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        # if not max_iters is None:
        #     n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
        #     self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]

        self.files = []

        if set_mask == 'masked':
            for name in self.img_ids:
                img_file = data_dir + name
                mask_file = data_dir + name.replace('img', 'mask').replace('image', 'mask')
                self.files.append({
                    'img': img_file,
                    'masked': mask_file,
                    'name': name
                })
        elif set_mask == 'unmasked':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        if self.set_mask == 'masked':
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            with h5py.File(datafiles['masked'], 'r') as hf:
                mask = hf['mask'][:]
            name = datafiles['name']

            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            image = np.asarray(image, np.float32)
            mask = np.asarray(mask, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            # if self.transform is not None:
            #   for i in range(len(self.mean)):
            #       image[i, :, :] = self.transform(image=image[i, :, :])['image']

            for i in range(len(self.mean)):
                image[i, :, :] -= self.mean[i]
                image[i, :, :] /= self.std[i]

            return image.copy(), mask.copy(), np.array(size), name

        else:
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            name = datafiles['name']

            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            # if self.transform is not None:
            #     for i in range(len(self.mean)):
            #         image[i, :, :] = self.transform(image=image[i, :, :])['image']

            for i in range(len(self.mean)):
                image[i, :, :] -= self.mean[i]
                image[i, :, :] /= self.std[i]

            return image.copy(), np.array(size), name


if __name__ == '__main__':
    train_dataset = LandslideDataSet(data_dir='./data/', list_path='./dataset/train.txt')
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True)

    channels_sum, channel_squared_sum = 0, 0
    num_batches = len(train_loader)
    for data, _, _, _ in train_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])

    mean = channels_sum / num_batches
    std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
    print(mean, std)
    # [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
    # [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
