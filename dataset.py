from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import torch.nn.functional as F

class XrayDataset(Dataset):
    def __init__(self, dataframe : pd.DataFrame, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.df['full_path'].values[idx]).convert('RGB')
        label = self.df['encoded_label'].values[idx]

        if self.transform:
            image = self.transform(image)

        data = {'image': image, 'label': label}

        return data


class Dataprep():
    def __init__(self,classes):
        self.classes = classes
        self.class_dict = dict()
        for i,class_name in enumerate(classes):
            self.class_dict[class_name] = i

    def one_hot_encoding(self, data):
        num_classes = len(self.classes)
        targets = data.split('|')
        encoded_data = torch.zeros(num_classes,dtype=torch.int64)
        for key in targets:
            if key == 'No Finding':
                return encoded_data
            encoded_data += F.one_hot(torch.tensor(int(self.class_dict[key]),dtype=torch.int64).squeeze(), num_classes=num_classes)
        encoded_data = encoded_data.to(torch.float)
        return encoded_data


    def prep(self, BASE_PATH, CSV_PATH, TRAIN_LIST_PATH, TEST_LIST_PATH, train_val_split_ratio = 0.1):
        df_all = pd.read_csv(CSV_PATH)
        df_all['full_path'] = BASE_PATH + df_all['Image Index']
        df_all['encoded_label'] = df_all['Finding Labels'].apply(self.one_hot_encoding)

        train_val_imgs = pd.read_csv(TRAIN_LIST_PATH).values.squeeze()
        test_imgs = pd.read_csv(TEST_LIST_PATH).values.squeeze()
        df_train = df_all[df_all['Image Index'].isin(train_val_imgs).values]
        df_test = df_all[df_all['Image Index'].isin(test_imgs).values]

        total_train_imgs = len(df_train)
        total_val_imgs = int(train_val_split_ratio * total_train_imgs)
        df_val = df_train[:total_val_imgs]
        df_train = df_train[total_val_imgs:]

        return df_train, df_val, df_test