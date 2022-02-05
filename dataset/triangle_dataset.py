import torch
import torchvision.transforms as transforms
from PIL import Image


class TriangleDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        if "random" in img_path:
            label = 0
        elif "equilateral" in img_path:
            label = 1

        return img, label
