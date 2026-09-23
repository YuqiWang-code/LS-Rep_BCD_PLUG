"""
Change Detection Dataset for A2Net

Supports: LEVIR-CD-256, SYSU-CD-256, WHU-CD-256, CDD-CD-256
"""

import cv2
import numpy as np
import torch.utils.data
import os


class CDDataset(torch.utils.data.Dataset):
    """
    Change Detection Dataset

    Returns 6-channel concatenated image: [T1(3ch), T2(3ch)]
    """

    def __init__(self, dataset, file_root='data/', transform=None, dataset_name='LEVIR'):
        """
        Args:
            dataset: 'train', 'val', 'test'
            file_root: dataset root directory
            dataset_name: 'LEVIR', 'SYSU', 'WHU', 'CDD'
            transform: data augmentation transforms
        """
        # Read file list
        list_path = os.path.join(file_root, 'list', dataset + '.txt')
        self.file_list = open(list_path).read().splitlines()

        # Build paths using os.path.join
        self.pre_images = [os.path.join(file_root, 'A', x) for x in self.file_list]
        self.post_images = [os.path.join(file_root, 'B', x) for x in self.file_list]
        self.gts = [os.path.join(file_root, 'label', x) for x in self.file_list]
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]

        # Read images
        pre_image = cv2.imread(pre_image_name)
        post_image = cv2.imread(post_image_name)
        label = cv2.imread(label_name, 0)

        # Concatenate dual-temporal images (6 channels)
        img = np.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            [img, label] = self.transform(img, label)

        return img, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}


def get_loader(file_root, list_file, img_ext='.png', file_prefix='',
               batchsize=32, trainsize=256, shuffle=True, num_workers=4, pin_memory=True,
               generator=None, worker_init_fn=None):
    """
    Build training dataloader with augmentation

    `generator` / `worker_init_fn` are forwarded verbatim to DataLoader so that
    the sample order (shuffle) and per-worker augmentation RNG are deterministic
    and independent of model-construction time RNG (e.g. GRAFT param init).
    """

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from transforms import Compose, Normalize, Scale, RandomCropResize, RandomFlip, RandomExchange, ToTensor

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    transform = Compose([
        Normalize(mean=mean, std=std),
        Scale(trainsize, trainsize),
        RandomCropResize(int(7. / 224. * trainsize)),
        RandomFlip(),
        RandomExchange(),
        ToTensor()
    ])

    # Extract dataset name from file_root
    dataset_name = 'LEVIR'
    if 'SYSU' in file_root:
        dataset_name = 'SYSU'
    elif 'WHU' in file_root:
        dataset_name = 'WHU'
    elif 'CDD' in file_root:
        dataset_name = 'CDD'

    # Determine split from list_file
    if 'train' in list_file:
        split = 'train'
    elif 'val' in list_file:
        split = 'val'
    else:
        split = 'test'

    dataset = CDDataset(split, file_root=file_root, transform=transform, dataset_name=dataset_name)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train'),
        generator=generator,
        worker_init_fn=worker_init_fn
    )

    return loader


def get_test_loader(file_root, list_file, img_ext='.png', file_prefix='',
                    batchsize=32, testsize=256, num_workers=4, pin_memory=True,
                    generator=None, worker_init_fn=None):
    """
    Build test/validation dataloader without augmentation
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from transforms import Compose, Normalize, Scale, ToTensor

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    transform = Compose([
        Normalize(mean=mean, std=std),
        Scale(testsize, testsize),
        ToTensor()
    ])

    # Extract dataset name from file_root
    dataset_name = 'LEVIR'
    if 'SYSU' in file_root:
        dataset_name = 'SYSU'
    elif 'WHU' in file_root:
        dataset_name = 'WHU'
    elif 'CDD' in file_root:
        dataset_name = 'CDD'

    # Determine split from list_file
    if 'val' in list_file:
        split = 'val'
    else:
        split = 'test'

    dataset = CDDataset(split, file_root=file_root, transform=transform, dataset_name=dataset_name)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn
    )

    return loader
