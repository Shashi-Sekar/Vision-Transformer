import torch

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt

to_tensor = [Resize((144, 144)), ToTensor()]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image
    
def show_images(images, num_samples=40, cols=8):
    plt.figure(figsize=(15, 15))
    index = int(len(dataset) / num_samples)

    print(images)

    for i, img in enumerate(images):
        if i % index == 0:
            plt.subplot(int(num_samples / cols) + 1, cols, int(i / index) + 1)
            plt.imshow(to_pil_image(img[0][0]))

dataset = OxfordIIITPet(root='.', download=True, transform=Compose(to_tensor))
#dataset = OxfordIIITPet(root='.', download=True)

#show_images(dataset)

print(f"Number of samples: {len(dataset)}")
print(f"Number of classes: {len(dataset.classes)}")

train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=True)