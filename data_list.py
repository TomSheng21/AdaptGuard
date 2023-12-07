import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision
import copy

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


# backdoor


class ImageList_Backdoor(Dataset):
    def __init__(self, image_list, y_target=None, poisoned_rate=None, test_backdoor_set=False, stay_target=False,
        labels=None, transform=None, target_transform=None, mode='RGB', backdoor='None'):

        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        # Blended
        self.add_img = 'hello_kitty.jpeg'
        self.alpha = 0.2
        # SIG
        self.delta = 20
        self.f = 6

        self.y_target = y_target
        self.poisoned_rate = poisoned_rate
        self.stay_target = stay_target
        self.test_backdoor_set = test_backdoor_set
        self.backdoor = backdoor

        if self.backdoor == 'None':
            pass
        else:
            if self.test_backdoor_set:
                assert poisoned_rate == 1.0
                imgs_length = len(self.imgs)
                self.imgs = [_ for _ in self.imgs if _[1] != self.y_target]
                print('re-generate Test Backdoor dataset, drop class ',self.y_target,'.')
                print(imgs_length,'->',len(self.imgs))
                total_num = len(self.imgs)
                self.poisoned_set = frozenset(list(range(total_num)))
            else:
                total_num = len(self.imgs)
                poisoned_num = int(total_num * self.poisoned_rate)
                assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
                tmp_list = list(range(total_num))
                random.shuffle(tmp_list)
                self.poisoned_set = frozenset(tmp_list[:poisoned_num])

            self.add_img_trans = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.ToTensor(),
                ])

        if self.backdoor == 'None':
            pass
        elif self.backdoor == 'Blended':         
            self.add_img_tensor = self.add_img_trans(rgb_loader(self.add_img))
        elif self.backdoor == 'SIG':
            sig_tensor = torch.zeros(3,256,256)
            for i in range(sig_tensor.shape[0]):
                for j in range(sig_tensor.shape[1]):
                    for k in range(sig_tensor.shape[2]):
                        sig_tensor[i][j][k] = (self.delta/255.0) * np.sin(2 * np.pi * k * self.f / sig_tensor.shape[2])
            self.sig_tensor = sig_tensor
        else:
            assert False


    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        
        if self.backdoor == 'Blended' and index in self.poisoned_set:
            img = self.add_img_trans(img) * (1-self.alpha) + self.add_img_tensor * self.alpha
            img = torchvision.transforms.ToPILImage()(img)
            if not self.stay_target:
                target = self.y_target

        elif self.backdoor == 'SIG' and index in self.poisoned_set:
            img = self.add_img_trans(img) + self.sig_tensor
            img = torch.clamp(img, 0, 1)
            img = torchvision.transforms.ToPILImage()(img)
            if not self.stay_target:
                target = self.y_target

        else:
            pass
        
        sample = self.transform(img)
        return sample, target
        

    def __len__(self):
        return len(self.imgs)


