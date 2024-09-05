import numpy as np
import os
import pickle
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
CLASS = ["apple","book","bowtie","candle","cup","door","envelope","eyeglasses","guitar","hammer","hat","ice_cream","leaf","pants","scissors","star","t-shirt"]
class Quickdrawdatasets(Dataset):
    def __init__(self, root,total_img_per_class, percent, mode):
        self.root = root
        self.num_class = 17
        if mode =="train":
            self.begin = 0
            self.num_image_per_class = int(total_img_per_class*percent)
        else:
            self.begin = int(total_img_per_class*percent)
            self.num_image_per_class = int(total_img_per_class*(1-percent))
        self.num_sample = self.num_image_per_class*self.num_class
    def __len__(self):
        return self.num_sample
    def __getitem__(self, item):
        file = "{}/full_numpy_bitmap_{}.npy".format(self.root,CLASS[int(item/self.num_image_per_class)])
        image = np.load(file).astype(np.float32)[self.begin+(item%self.num_image_per_class)]
        image = image/255
        image = image.reshape((1,28,28))
        label = int(item/self.num_image_per_class)
        return image, label
if __name__ == '__main__':
    dataset = Quickdrawdatasets(root = "data_quick_draw", total_img_per_class=500, percent=0.8, mode="train")
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = 16,
        num_workers = 4,
        drop_last = True,
        shuffle=True
    )
    for images, labels in dataloader:
        print(images.shape, labels)