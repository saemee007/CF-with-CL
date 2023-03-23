import os
import sys
from PIL import Image

import torchvision.transforms as transforms

from data.noisy_cifar import NoisyCIFAR10



class CIFAR10Contrastive(NoisyCIFAR10):
    # 그냥 이미지 받아서 한 이미지 당 moco_aug를 두 번 적용해서 두 이미지 return
    def __init__(self,
                 aug=None, 
                 root='/home/saemeechoi/cls_noise/knn_analysis/data/datasets/cifar10', 
                 train=True,
                 noise_type=None,
                 openset_ratio=0.0,
                 closeset_ratio=0.0,
                 return_label=False
                 ):
        super().__init__(root=root, train=train, noise_type=noise_type, openset_ratio=openset_ratio, closeset_ratio=closeset_ratio) # , noise_type='symmetric'
        self.train = train
        self.return_label=return_label
        if aug is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = aug

    def __len__(self):
        return self.train_data.__len__()

    def __getitem__(self, index):
        if self.train:
            img, noisy_target, target = self.train_data[index], self.train_noisy_labels[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        
        if self.return_label:
            if self.train:
                return self.transform(img), noisy_target, target
            return self.transform(img), target
            
        if isinstance(self.transform, list):
            return self.transform[0](img), self.transform[1](img)
        return self.transform(img), self.transform(img)

# TODO
# class CIFAR(BaseDataset):
#     # 각 이미지에 moco_aug 적용해서 label가 함께 return
#     def __init__(self, mode='train', max_class=1000, aug=None):
#         super().__init__(mode, max_class, aug)

#     def __len__(self):
#         return self.samples.__len__()

#     def __getitem__(self, index):
#         label, name = self.samples[index]
#         filename = os.path.join(self.image_folder, name)
#         img = self.load_image(filename)
#         return self.transform(img), label




