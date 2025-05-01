import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision.transforms.functional as F
import numbers
import numpy as np
import datetime

from utils.transforms import ToClassLabels
from utils.transforms import NewPad

from train import Trainer
from model import FastSCNN
from utils.dataset import PostdamDataset, UDD, OverwatchDataset
from metrics import pixel_accuracy

num_epochs = 40
batch_size = 32
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ClassesColors = {
    (255, 255, 255): 0, # impervious_surfaces
    (0, 0, 255): 1, # building
    (0, 255, 255): 2, # low_vegetation
    (0, 255, 0): 3, # tree
    (255, 255, 0): 4, # car
    (255, 0, 0): 5 # background
    }

def preprocessing(image, mask):
    mask_transformer = transforms.Compose([
        NewPad(fill=(255,0,0)),
        ToClassLabels()
#         transforms.Lambda(lambda x: to_class_labels(x))
    ])
    image_transformer = transforms.Compose([
        NewPad(),
        transforms.ToTensor(),
        transforms.Normalize([0.3396, 0.3628, 0.3362], [0.1315, 0.1287, 0.1333])
    ])
    return image_transformer(image).float(), mask_transformer(mask)

def UDD_preprocessing(image, mask):
    mask = np.array(mask).astype('int64')
    mask = torch.torch.from_numpy(mask)
    image_transformer = transforms.Compose([
        NewPad(),
        transforms.ToTensor(),
        transforms.Normalize([0.3967, 0.4193, 0.4018], [0.1837, 0.1673, 0.1833])
    ])
    return image_transformer(image).float(), mask

# Define transforms for images and labels
def Overwatch_processing(image, label):
    image_transformer = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2555], std=[0.1917])  # Use computed values
    ])

    label = label.resize((512, 512))  # Resize to match image size
    #label = np.array(label).astype('int64')
    #label = torch.torch.from_numpy(label)
    label = transforms.ToTensor()(label)  # Convert to tensor

    return image_transformer(image).float(), label.squeeze(0).long()

# Transform for labels (grayscale, categorical, no normalization)
def label_transform(label):
    label = label.resize((512, 512))  # Resize to match image size
    #label = np.array(label).astype('int64')
    #label = torch.torch.from_numpy(label)
    label = transforms.ToTensor()(label)  # Convert to tensor
    return label.squeeze(0).long() 

train_image_path = 'D:/IR Data Comparison/Extracted_Overwatch_Data/Good_data/train/images/'
train_label_path = 'D:/IR Data Comparison/Extracted_Overwatch_Data/Good_data/train/labels/'
test_image_path = 'D:/IR Data Comparison/Extracted_Overwatch_Data/Good_data/Val/Data/images/'
test_label_path = 'D:/IR Data Comparison/Extracted_Overwatch_Data/Good_data/Val/Data/labels/'
ds_train = OverwatchDataset(train_image_path, train_label_path, transform=Overwatch_processing)
ds_test = OverwatchDataset(test_image_path, test_label_path, transform=Overwatch_processing)

dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_test = DataLoader(ds_test, batch_size, shuffle=False)

model = FastSCNN(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()
success_metric = pixel_accuracy
trainer = Trainer(model, criterion, optimizer, success_metric, device, None)
fit_res = trainer.fit(dl_train,
                      dl_test,
                      num_epochs= num_epochs,
                      checkpoints='checkpoints/' + model.__class__.__name__ + datetime.datetime.today().strftime("%m_%d"))

print(fit_res)
