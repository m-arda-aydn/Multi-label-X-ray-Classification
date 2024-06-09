import numpy as np 
import torch.nn as nn
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
import time 
from datetime import datetime
from torchvision.models import vit_b_16, convnext_base, swin_v2_b, densenet121, maxvit_t, efficientnet_b4
import os
from dataset import XrayDataset, Dataprep
from torchmetrics.classification import MultilabelAUROC
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from utils import return_seconds

classes = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

BASE_PATH = '/mnt/disk2/comp_aided/images/'
CSV_PATH = '/mnt/disk2/comp_aided/Data_Entry_2017_v2020.csv'
TRAIN_LIST_PATH = '/mnt/disk2/comp_aided/train_val_list.txt'
TEST_LIST_PATH = '/mnt/disk2/comp_aided/test_list.txt'

df_train, df_val, df_test = Dataprep(classes).prep(BASE_PATH=BASE_PATH, CSV_PATH=CSV_PATH, 
                        TRAIN_LIST_PATH=TRAIN_LIST_PATH, TEST_LIST_PATH=TEST_LIST_PATH, train_val_split_ratio=0.1)

print(df_train.shape)
print(df_val.shape)


IMG_SIZE = 224

train_transform = T.Compose([T.Resize(IMG_SIZE),
                        T.ToTensor(),
                        T.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD), # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(),
                        T.RandomRotation(degrees=10)])

test_transform = T.Compose([T.Resize(IMG_SIZE),
                        T.ToTensor(),
                        T.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)]) # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

# model = vit_b_16()
# model.heads = nn.Sequential(nn.Linear(768,len(classes),bias=True))
model = densenet121(weights = 'IMAGENET1K_V1')
model.classifier = nn.Linear(in_features=1024,out_features=len(classes),bias=True)
# model = maxvit_t(weights='IMAGENET1K_V1')
# model.classifier[-1] = nn.Linear(in_features=512,out_features=len(classes),bias=False)
# model = efficientnet_b4(weights='IMAGENET1K_V1')
# model.classifier[-1] = nn.Linear(in_features=1792,out_features=len(classes),bias=True)

model_type_name = model.__class__.__name__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
VAL_BATCH_SIZE = 512

train_set = XrayDataset(df_train,transform=train_transform)
trainloader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)

val_set = XrayDataset(df_val,transform=test_transform)
valloader = DataLoader(val_set,batch_size=VAL_BATCH_SIZE,shuffle=False)

WORK_DIR = '/mnt/disk2/comp_aided/work_dir'
CHECKPOINT_DIR = '/mnt/disk2/comp_aided/checkpoints/'
EPOCH = 60
lr = 1e-3
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
T_0 = 20
T_mult = 1
eta_min = 5e-5
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult,eta_min=eta_min)
print_loss_every_iter = 50
model.to(device)
model.train()
init_true_threshold = 0.75
# true_threshold = torch.nn.Parameter(torch.tensor([init_true_threshold],dtype=torch.float,device=device))
true_threshold = init_true_threshold 
best_acc = -1
best_auc = -1
now = datetime.now()
filename_time = now.strftime("%d_%m_%Y_%H_%M_%S")
write_every = 5
auc_metric = MultilabelAUROC(num_labels=len(classes),average="macro", thresholds=None).to(device)
dir_name = filename_time + '_' + model_type_name

try:
    os.mkdir(CHECKPOINT_DIR + dir_name)
except FileExistsError:
    print(f'Directory: {CHECKPOINT_DIR + dir_name} already exists')

with open(os.path.join(WORK_DIR,filename_time + '_' + model_type_name + '_score.txt'),'a') as txt_file:
    txt_file.write(f'Model Name: {model_type_name}\n' 
                   f'Initial learning rate: {lr}\n' 
                   f'loss type: {criterion.__class__.__name__}\n'
                   f'batch size: {BATCH_SIZE}\n' 
                   f'Initial true threshold: {init_true_threshold}\n'
                   f'scheduler: {scheduler.__class__.__name__}\n')

for epoch in range(EPOCH):

    print(f'Epoch: {epoch + 1} / {EPOCH} ...')
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        model.train()
        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        labels = labels.to(torch.float)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if i % print_loss_every_iter == print_loss_every_iter - 1:    
            print(f'loss in {i + 1} / {len(trainloader)} iteration for {epoch + 1} / {EPOCH} epoch : {running_loss / print_loss_every_iter:.3f}')
            with open(os.path.join(WORK_DIR,filename_time + '_' + model_type_name + '_score.txt'),'a') as txt_file:
                txt_file.write(f'loss in {i + 1} / {len(trainloader)} iteration for {epoch + 1} / {EPOCH} epoch : {running_loss / print_loss_every_iter:.3f}\n')
            
            running_loss = 0.0
    number_true_pred = 0
    auc_list = []

    for i, data in enumerate(valloader):
        print(f'Starting validation for epoch: {epoch + 1} / {EPOCH}')
        batch_true_pred = 0
        model.eval()
        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        labels = labels.to(torch.int)

        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            auc = auc_metric(outputs,labels).item()
            auc_list.append(auc)
            preds = (outputs >= true_threshold).int()

            for b in range(outputs.shape[0]):
                if torch.all(preds[b,:] == labels[b,:]).item():
                    number_true_pred += 1
                    batch_true_pred += 1
        
        print(f'Validation accuracy in {i + 1} / {len(valloader)} batch for {epoch + 1} / {EPOCH} epoch: {batch_true_pred / VAL_BATCH_SIZE:.3f}')
        print(f'Area Under Curve (AUC) in {i + 1} / {len(valloader)} batch for {epoch + 1} / {EPOCH} epoch: {auc:.3f}')

        if i % write_every == write_every - 1:
            with open(os.path.join(WORK_DIR,filename_time + '_' + model_type_name + '_score.txt'),'a') as txt_file:
                txt_file.write(f'Validation accuracy in {i + 1} / {len(valloader)} batch for {epoch + 1} / {EPOCH} epoch: {batch_true_pred / VAL_BATCH_SIZE:.3f}\n')
                txt_file.write(f'Area Under Curve (AUC) in {i + 1} / {len(valloader)} batch for {epoch + 1} / {EPOCH} epoch: {auc:.3f}\n')
    
    print('Validation finished...')
    accuracy = number_true_pred / len(val_set)
    mean_auc = sum(auc_list) / len(auc_list)
    if accuracy > best_acc or mean_auc > best_auc:
        if accuracy > best_acc:
            best_acc = accuracy
        if mean_auc > best_auc:
            best_auc = mean_auc
        print('Saving the best model checkpoint...')
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, dir_name, filename_time + '_' + model_type_name + '_epoch_' + str(epoch + 1)  + '.pth'))
    print(f'Validation accuracy for (epoch : {epoch + 1} / {EPOCH}): {accuracy:.3f}, best accuracy: {best_acc}')
    print(f'Area Under Curve (AUC) for (epoch : {epoch + 1} / {EPOCH}): {mean_auc:.3f}, best AUC: {best_auc}')
    
    end_time = time.time()
    epoch_time = (end_time - start_time)
    epoch_time_dict = return_seconds(epoch_time)
    estimated_time_dict = return_seconds((epoch_time * (EPOCH - epoch - 1)))
    print(f'Time passed during this epoch ({epoch + 1} / {EPOCH}) : ',end='')
    for key, value in epoch_time_dict.items():
        print(f'{value} {key}',end=' ')
    print()
    print(f'Estimated time to complete training : ', end='')
    for key, value in estimated_time_dict.items():
        print(f'{value} {key}',end=' ')
    print()  

    with open(os.path.join(WORK_DIR,filename_time + '_' + model_type_name + '_score.txt'),'a') as txt_file:
        txt_file.write(f'Validation accuracy for (epoch : {epoch + 1} / {EPOCH}): {accuracy:.3f}, best accuracy: {best_acc}\n')
        txt_file.write(f'Area Under Curve (AUC) for (epoch : {epoch + 1} / {EPOCH}): {mean_auc:.3f}, best AUC: {best_auc}\n')
        txt_file.write(f'Time passed during this epoch ({epoch + 1} / {EPOCH}) : ')
        for key, value in epoch_time_dict.items():
            txt_file.write(f'{value} {key} ')
        txt_file.write(f'\nEstimated time to complete training : ')
        for key, value in estimated_time_dict.items():
            txt_file.write(f'{value} {key} ')
        txt_file.write('\n')

print('Training finished')
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, dir_name, filename_time + '_' + model_type_name + '_epoch_' + str(epoch + 1)  + '.pth'))
