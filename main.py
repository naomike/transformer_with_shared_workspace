import glob
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.seed_func import fix_seed
from dataset.triangle_dataset import TriangleDataset
from model import EasyTransformer


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--with_attention', action='store_false')
parser.add_argument('--with_shared_workspace', action='store_false')
parser.add_argument('--train_only_shared_workspace', action='store_true')
args = parser.parse_args()

SEED = 42
fix_seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# create datasets and dataloaders
phases = ['train', 'valid', 'test']
dataset_dict = {'train': [], 'valid': [], 'test': []}
dataloader_dict = {'train': [], 'valid': [], 'test': []}
for phase in phases:
    img_paths = []
    for img_path in glob.glob('{}/{}/*.png'.format(args.data_dir, phase)):
        img_paths.append(img_path)
    dataset_dict[phase] = TriangleDataset(img_paths=img_paths)
    if phase == 'train':
        dataloader_dict[phase] = \
            torch.utils.data.DataLoader(dataset_dict[phase], batch_size=32, shuffle=True)
    else:
        dataloader_dict[phase] = \
            torch.utils.data.DataLoader(dataset_dict[phase], batch_size=1, shuffle=False)


model = EasyTransformer(
    device=device, with_attention=args.with_attention, \
    with_shared_workspace=args.with_shared_workspace, img_size=64
).to(device)
if args.train_only_shared_workspace:
    params_to_update = []
    for name, param in model.named_parameters():
        if 'shared_workspace' in name:
            param.requires_grad = True
            params_to_update.append(param)
        elif 'pos' not in name:
            param.requires_grad = False
else:
    params_to_update = model.parameters()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
epochs = 100
best_valid_acc = 0

for epoch in range(0, epochs):
    epoch_train_loss = 0
    epoch_train_acc = 0
    epoch_valid_loss = 0
    epoch_valid_acc = 0
    # train
    model.train()
    for (inputs, labels) in dataloader_dict['train']:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += float(loss)
        epoch_train_acc += (outputs.argmax(dim=1) == labels).float().sum()
    epoch_train_loss /= len(dataset_dict['train'])
    epoch_train_acc /= len(dataset_dict['train'])

    # valid
    model.eval()
    with torch.no_grad():
        for (inputs, labels) in dataloader_dict['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_valid_loss += float(loss)
            epoch_valid_acc += (outputs.argmax(dim=1) == labels).float().sum()
    epoch_valid_loss /= len(dataset_dict['valid'])
    epoch_valid_acc /= len(dataset_dict['valid'])

    scheduler.step()
    if epoch_valid_acc > best_valid_acc:
        best_valid_acc = epoch_valid_acc
        torch.save(model.state_dict(), 'model.pth')
    print('-----Epoch {}-----'.format(epoch+1))
    print(f'train acc: {epoch_train_acc:.3f}, train loss: {epoch_train_loss:.3f}')
    print(f'valid acc: {epoch_valid_acc:.3f}, valid loss: {epoch_valid_loss:.3f}')


# test
model.load_state_dict(torch.load('model.pth'))
model.eval()
test_acc = 0
test_loss = 0
with torch.no_grad():
    for (inputs, labels) in dataloader_dict['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += float(loss)
        test_acc += (outputs.argmax(dim=1) == labels).float().sum()
    test_loss /= len(dataset_dict['test'])
    test_acc /= len(dataset_dict['test'])

    print(f'test acc: {test_acc:.3f}')
    print(f'test loss: {test_loss:.3f}')
