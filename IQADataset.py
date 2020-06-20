import torch
import os
from torch.utils.data import Dataset
from scipy.signal import convolve2d
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor

def gray_loader(path):
    return Image.open(path).convert('L')

def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln

def CropPatches(image, patch_size=32, stride=32):
    w, h = image.size
    patches = ()
    for i in range(0, h-stride, stride):
        for j in range(0, w-stride, stride):
            patch = to_tensor(image.crop((j, i, j+patch_size, i+patch_size)))
            patch = LocalNormalization(patch[0].numpy())
            patches = patches + (patch,)
    return patches


class IQADataset(Dataset):
    def __init__(self, dataset, config, index, status):
        self.loader = gray_loader
        im_dir = config[dataset]['im_dir']
        self.patch_size = config['patch_size']
        self.stride = config['stride']

        test_ratio = config['test_ratio']
        train_ratio = config['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []

        ref_ids = []
        for line0 in open("./data/ref_ids.txt", "r"):
            line0 = float(line0[:-1])
            ref_ids.append(line0)
        ref_ids = np.array(ref_ids)
        # ref_ids = ref_ids[0:10]

        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = []
        for line5 in open("./data/mos.txt", "r"):
            line5 = float(line5.strip())
            self.mos.append(line5)
        self.mos = np.array(self.mos)

        im_names = []
        ref_names = []
        for line1 in open("./data/im_names.txt", "r"):
            line1 = line1.strip()
            im_names.append(line1)
        im_names = np.array(im_names)
        # print(im_names)

        for line2 in open("./data/refnames.txt", "r"):
            line2 = line2.strip()
            ref_names.append(line2)
        ref_names = np.array(ref_names)

        self.patches = ()
        self.label = []

        self.im_names = [im_names[i] for i in self.index]
        self.ref_names = [ref_names[i] for i in self.index]
        self.mos = [self.mos[i] for i in self.index]

        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(self.im_names[idx]))
            im = self.loader(os.path.join(im_dir, self.im_names[idx]))
            patches = CropPatches(im, self.patch_size, self.stride)

            if status == 'train':
                self.patches = self.patches + patches
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
            else:
                self.patches = self.patches + (torch.stack(patches), )
                self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], (torch.Tensor([self.label[idx]]))




