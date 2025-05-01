import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import glob
import sys
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

class ClassesColors:
    impervious_surfaces = (255, 255, 255)
    building = (0, 0, 255)
    low_vegetation = (0, 255, 255)
    tree = (0, 255, 0)
    car = (255, 255, 0)
    background = (255, 0, 0)

    def __len__(self):
        return 3

class OverwatchDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])
        image = Image.open(img_path).convert("L") # converting all images to Grayscale
        label = Image.open(label_path).convert("L")  # Assuming label is grayscale

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

class PostdamDataset(Dataset):
    def __init__(self, images_path, label_path, transform=None, load_tensor=False):
        self.images = glob.glob(os.path.join(images_path, '*.tif'))
        self.labels_path = label_path
        self.transform = transform
        self.load_tensor = load_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < 0 or idx >len(self.images):
            raise IndexError("Index out of bound")

        image_path = self.images[idx]
        image_name =  image_path.split('/')[-1]
        img = Image.open(image_path)
        
        label_name = image_name.replace('RGB', 'label')
        label_path = os.path.join(self.labels_path, label_name)
        if self.load_tensor:
            label_path = label_path.replace('.tif', '.pt')
            label = torch.load(label_path)
        else:
            label = Image.open(label_path)

        if self.transform:
            img, label = self.transform(img, label)
        else:
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
            label = to_tensor(label)

        return img, label


class UDD(Dataset):
    def __init__(self, images_path, label_path, transform=None):
        self.images = glob.glob(os.path.join(images_path, '*.png'))
        self.labels_path = label_path
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < 0 or idx > len(self.images):
            raise IndexError("Index out of bound")

        image_path = self.images[idx]
        image_name = image_path.split('/')[-1]
        img = Image.open(image_path)

        label_path = os.path.join(self.labels_path, image_name)
        label = Image.open(label_path)
        if self.transform:
            img, label = self.transform(img, label)
        else:
            to_tensor = transforms.ToTensor()
            img, label = to_tensor(img), to_tensor(label)

        return img, label


class AirSimDataset(Dataset):
    """A class for using the aeroscapes data set."""
    def __init__(self, root: str, partition: str = 'train', transforms=None):
        """
        Initialize the data set.
        Args:
            root: the root directory of the data set
            transforms: a transformation that takes input PIL images and returns
        """
        super().__init__()
        if partition not in ['train', 'test']:
            raise ValueError("partition should be either 'train' or 'test', got ", partition)
        if not os.path.isdir(root):
            raise NotADirectoryError(f'{root} is not a directory.')
        self.transforms = transforms
        # retrieve the set of id's for the current partition
        # collect all the file names
        images_dir = os.path.join(root, 'images')
        masks_dir = os.path.join(root, 'masks')
        if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
            raise NotADirectoryError(f'{root} is does not contain \'images\' and \'masks\' subdirectories.')

        self.n_samples = len(os.listdir(images_dir))
        if self.n_samples <= 0:
            raise FileNotFoundError('sub directories should not be empty.')
        train_size = int(0.8 * self.n_samples)
        if partition == 'train':
            self.n_samples = train_size
            self.images = [os.path.join(images_dir, f'{i}.png') for i in range(train_size)]
            self.masks = [os.path.join(masks_dir, f'{i}.png') for i in range(train_size)]
        else:
            self.images = [os.path.join(images_dir, f'{i}.png') for i in range(train_size, self.n_samples)]
            self.masks = [os.path.join(masks_dir, f'{i}.png') for i in range(train_size, self.n_samples)]
            self.n_samples = self.n_samples - train_size

        assert len(self.images) == len(self.masks), 'Error: images and masks should be of the same size.'

    def __getitem__(self, index):
        if index >= self.n_samples:
            raise IndexError(f'{index} is out of range, only {self.n_samples} samples.')
        img = Image.open(self.images[index])
        mask = Image.open(self.masks[index])

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        else:
            to_tensor = transforms.ToTensor()
            img, mask = to_tensor(img), to_tensor(mask)

        return img, mask

    def __len__(self):
        return self.n_samples

def compute_mean_std(images_path, label_path):
    dataset = UDD(images_path, label_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for i, data in enumerate(loader):
        data = data[0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(mean)
    print(std)

if __name__ == "__main__":
    compute_mean_std(sys.argv[1], sys.argv[2])