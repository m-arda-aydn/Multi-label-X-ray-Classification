import numpy as np 
import torch.nn as nn
import torch
from torchvision import transforms as T
import os
from torchvision.models import vit_b_16
from dataset import XrayDataset,Dataprep
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC, MultilabelAccuracy

classes = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

BASE_PATH = '/mnt/disk2/comp_aided/images/'
CSV_PATH = '/mnt/disk2/comp_aided/Data_Entry_2017_v2020.csv'
TRAIN_LIST_PATH = '/mnt/disk2/comp_aided/train_val_list.txt'
TEST_LIST_PATH = '/mnt/disk2/comp_aided/test_list.txt'

_, _ , df_test = Dataprep(classes).prep(BASE_PATH=BASE_PATH, CSV_PATH=CSV_PATH, 
                                                   TRAIN_LIST_PATH=TRAIN_LIST_PATH, TEST_LIST_PATH=TEST_LIST_PATH)

IMG_SIZE = 224
test_transform = T.Compose([T.Resize(IMG_SIZE),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
print(BATCH_SIZE)
test_set = XrayDataset(df_test,transform=test_transform)
testloader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)

model = vit_b_16()
model.heads = nn.Sequential(nn.Linear(768,len(classes)))
model.load_state_dict(torch.load('/mnt/disk2/comp_aided/checkpoints/29_04_2024_02_13_02_VisionTransformer_epoch_1.pth'))
model.eval()
model.to(device)
true_threshold = 0.75 
auc_metric = MultilabelAUROC(num_labels=len(classes),average="macro", thresholds=None).to(device)
acc_metric = MultilabelAccuracy(num_labels=len(classes),threshold=true_threshold).to(device)

for i, data in enumerate(testloader):
        batch_true_pred = 0
        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        labels = labels.to(torch.int)

        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            print('Accuracy: ', acc_metric(outputs,labels))
            print('Area Under Curve: ', auc_metric(outputs,labels).item())
            preds = (outputs >= true_threshold).int()
            print('prediction: ', preds.tolist())
            print('Ground Truth: ', labels.tolist())
