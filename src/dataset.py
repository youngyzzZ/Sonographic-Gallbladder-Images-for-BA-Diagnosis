# @ FileName: dataset.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
import os
from os.path import join
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src import gallbladder_dataset


class BaseLoader:
    def __init__(self, args):
        # special params
        self.num_workers = args['num_workers']
        if 'batch_size' in args:
            self.batch_size = args['batch_size']

        # custom properties
        self._dataset_train = None
        self._dataset_eval = None
        self._dataset_test = None
        self._dataset_norm = None
        self._dataset_attack = None
        self.dataloader_train = None
        self.dataloader_eval = None
        self.dataloader_test = None

    @property
    def dataset_train(self):
        raise NotImplementedError

    @property
    def dataset_eval(self):
        raise NotImplementedError

    @property
    def dataset_test(self):
        raise NotImplementedError

    @property
    def dataset_norm(self):
        raise NotImplementedError

    @property
    def train(self):
        if self.dataloader_train is None:
            self.dataloader_train = Data.DataLoader(self.dataset_train,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers,
                                                    pin_memory=True)
        return self.dataloader_train

    @property
    def eval(self):
        if self.dataloader_eval is None:
            self.dataloader_eval = Data.DataLoader(self.dataset_eval,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True)
        return self.dataloader_eval

    @property
    def test(self):
        if self.dataloader_test is None:
            self.dataloader_test = Data.DataLoader(self.dataset_test,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True)
        return self.dataloader_test

    def cal_norm(self, n=1):
        return None

    @staticmethod
    def random_sample_base(base_dir, transform, size):
        classes = os.listdir(base_dir)
        classes.sort()
        images = []
        for c in classes:
            folder = join(base_dir, c)
            for file_name in np.random.choice(os.listdir(folder), size, False):
                img = join(folder, file_name)
                images.append(transform(Image.open(img)))
        return classes, torch.stack(images)


class CIFAR10(BaseLoader):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    num_classes = 10
    class_names = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, args):
        super(CIFAR10, self).__init__(args)
        self.base_dir = args['CIFAR10_dir']
        self.mean = CIFAR10.mean
        self.std = CIFAR10.std
        self.class_names = CIFAR10.class_names

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
            self._dataset_train = torchvision.datasets.CIFAR10(
                root=self.base_dir, train=True, transform=tf, download=True)
        return self._dataset_train

    @property
    def dataset_eval(self):
        if self._dataset_eval is None:
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)])
            self._dataset_eval = torchvision.datasets.CIFAR10(
                root=self.base_dir, train=False, transform=tf)
        return self._dataset_eval

    @property
    def dataset_norm(self):
        if self._dataset_norm is None:
            self._dataset_norm = torchvision.datasets.CIFAR10(
                self.base_dir, transform=transforms.ToTensor())
        return self._dataset_norm


class ImageNet100(BaseLoader):
    mean = [0.47881872, 0.45927624, 0.41515172]
    std = [0.27191086, 0.26549916, 0.27758414]
    num_classes = 100
    class_names = [str(i) for i in range(num_classes)]

    def __init__(self, args):
        super(ImageNet100, self).__init__(args)
        # base image dir
        self.base_dir = args['ImageNet100_dir']
        self.mean = ImageNet100.mean
        self.std = ImageNet100.std
        self.img_size = args['img_size']

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            tf = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                        scale=[0.7, 1.3]),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_train = torchvision.datasets.ImageFolder(
                join(self.base_dir, 'train'), transform=tf)
        return self._dataset_train

    @property
    def dataset_eval(self):
        if self._dataset_eval is None:
            tf = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_eval = torchvision.datasets.ImageFolder(
                join(self.base_dir, 'val'), transform=tf)
        return self._dataset_eval

    @property
    def dataset_norm(self):
        if self._dataset_norm is None:
            tf = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor()
            ])
            self._dataset_norm = ImageFolder(join(self.base_dir, 'train'),
                                             transform=tf)
        return self._dataset_norm

    def cal_norm(self, n=10):
        ds = self.dataset_norm
        dl = Data.DataLoader(dataset=ds, batch_size=self.batch_size,
                             num_workers=self.num_workers, pin_memory=True)
        m1 = m2 = m3 = s1 = s2 = s3 = 0
        for i in range(n):
            print('times:{}'.format(i))
            ll = len(dl)
            for idx, (x, _) in enumerate(dl):
                print('iter {} of {}'.format(idx, ll))
                x = x.cuda(non_blocking=True)
                m1 += x[:, 0, :, :].mean().item()
                m2 += x[:, 1, :, :].mean().item()
                m3 += x[:, 2, :, :].mean().item()
                s1 += x[:, 0, :, :].std().item()
                s2 += x[:, 1, :, :].std().item()
                s3 += x[:, 2, :, :].std().item()

        n = n * len(dl)
        print('mean: ', m1 / n, m2 / n, m3 / n)
        print('std: ', s1 / n, s2 / n, s3 / n)

    def random_sample(self, size):
        base_dir = join(self.base_dir, 'eval')
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ])
        return self.random_sample_base(base_dir, transform, size)


def get_dataloader(args):
    if args['dataset'] == 'ImageNet100':
        return ImageNet100(args)
    if args['dataset'] == 'CIFAR10':
        return CIFAR10(args)
    if args['dataset'] == 'Gallbladder':
        return Gallbladder(args)
    else:
        raise ValueError('No dataset: {}'.format(args['dataset']))


class Gallbladder(BaseLoader):
    # contain empty
    mean = [0.359, 0.361, 0.379]
    std = [0.190, 0.190, 0.199]
    # except empty
    # mean = [0.359, 0.361, 0.380]
    # std = [0.191, 0.190, 0.200]

    num_classes = 2
    class_names = ('Biliary atresia', 'Non-biliary atresia')

    def __init__(self, args):
        super(Gallbladder, self).__init__(args)
        self.train_dir = args['Gallbladder_train_dir']
        self.test_dir = args['Gallbladder_test_dir']
        self.train_csv = args['train_csv']
        self.test_csv = args['test_csv']
        self.mean = Gallbladder.mean
        self.std = Gallbladder.std
        self.img_size = args['img_size']
        self.convert = transforms.Grayscale(3)

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            tf = transforms.Compose([
                # transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                # transforms.RandomRotation([-180, 180]),
                # transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                #                         scale=[0.7, 1.3]),
                # transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_train = gallbladder_dataset.GallbladderDataset(csv_file=join(self.train_dir, self.train_csv),
                                                                         root_dir=self.train_dir,
                                                                         transform=tf)
        return self._dataset_train

    @property
    def dataset_eval(self):
        if self._dataset_eval is None:
            tf = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                self.gray_to_rgb,
                # transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            self._dataset_eval = gallbladder_dataset.GallbladderDataset(csv_file=join(self.test_dir, self.test_csv),
                                                                        root_dir=self.test_dir,
                                                                        transform=tf)
        return self._dataset_eval

    def gray_to_rgb(self, img):
        size = np.array(img).transpose().shape
        # print(size)
        if size[0] != 3:
            img = self.convert(img)

        return img


# if __name__ == '__main__':
#     args = {'dataset': 'ImageNet100',
#             'ImageNet100_dir': '/data/DataSets/MyImagenet',
#             'num_workers': 4,
#             'batch_size': 256}
#     ds = get_dataloader(args)
#     print('start cal')
#     ds.cal_norm(n=3)


if __name__ == '__main__':
    args = {'dataset': 'Gallbladder',
            'Gallbladder_train_dir': '/home/cccc/Desktop/share/deeplearning_project/pnasnet/data/DataSets/final_train',
            'Gallbladder_test_dir': '/home/cccc/Desktop/share/deeplearning_project/pnasnet/data/DataSets/final_train',
            'train_csv': './label_contain_empty/label_contain_empty_all.csv',
            'test_csv': './label_contain_empty/label.csv',
            'num_workers': 4,
            'batch_size': 16,
            'img_size': 331}  # pnasnet input size=331
    dl = get_dataloader(args).train
    # dl1 = get_dataloader(args).eval
    # for i, item in enumerate(dl):
    #     print(item)
    print(len(dl))

# if __name__ == '__main__':
#     args = {'dataset': 'CIFAR10',
#             'CIFAR10_dir': '/home/cccc/Desktop/share/open_data_set/cifar10',
#             'num_workers': 4,
#             'batch_size': 16,
#             'img_size': 32}
#     dl = get_dataloader(args).train
#     print(dl)
