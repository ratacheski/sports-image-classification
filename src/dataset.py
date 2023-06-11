import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# batch size
BATCH_SIZE = 256

# the training transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# the validation transforms
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder("./inputs/train",
                                                 transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder("./inputs/test",
                                                transform=valid_transform)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)
