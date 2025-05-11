from torch.utils.data import Dataset
import torch
import PIL


class Dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.data.loc[idx, "img"]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.data.loc[idx, "emotion_label"])
        return image, label
