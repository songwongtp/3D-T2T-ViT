from datasets.adni_dataset import ADNIDataset
from datasets.adni_transforms import ToTensor
from torchvision import transforms

data = '/mnt/ssd/songwong/Dataset/ADNI2'

dataset_train = ADNIDataset(data, 'train', transform=transforms.Compose([ToTensor()]))
dataset_val = ADNIDataset(data, 'val', transform=transforms.Compose([ToTensor()]))
