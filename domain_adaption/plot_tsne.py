import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.stats import trim_mean
import matplotlib.pyplot as plt
from util import EEGDataset
import torch.utils.data as Data


# 绘制特征散点图
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = 'E:/damo/data_sample/data_sample/'

train_psd_features = np.load(path+'source/patient_psd.npy', allow_pickle=True)
train_de_features = np.load(path+'source/patient_de.npy', allow_pickle=True)
train_time_features = np.load(path+'source/patient_time.npy', allow_pickle=True)
train_age_and_sex = np.load(path+'source/patient_age_and_sex.npy', allow_pickle=True)

test_psd_features = np.load(path+'target/patient_psd.npy', allow_pickle=True)
test_de_features = np.load(path+'target/patient_de.npy', allow_pickle=True)
test_time_features = np.load(path+'target/patient_time.npy', allow_pickle=True)
test_age_and_sex = np.load(path+'target/patient_age_and_sex.npy', allow_pickle=True)

train_x = np.concatenate((train_psd_features.reshape(len(train_psd_features), -1),
                          train_de_features.reshape(len(train_de_features), -1),
                          train_time_features.reshape(len(train_time_features), -1),
                          train_age_and_sex.reshape(len(train_age_and_sex), -1)), axis=-1)
train_label = np.load(path+'source/patient_label.npy', allow_pickle=True).reshape(-1)
train_label[train_label == 3] = 0
print(pd.value_counts(train_label))

test_x = np.concatenate((test_psd_features.reshape(len(test_psd_features), -1),
                         test_de_features.reshape(len(test_de_features), -1),
                         test_time_features.reshape(len(test_time_features), -1),
                         test_age_and_sex.reshape(len(test_age_and_sex), -1)), axis=-1)
test_label = np.load(path+'target/patient_label.npy', allow_pickle=True).reshape(-1)

test_label[test_label == 3] = 0
print(pd.value_counts(test_label))

train_dataset = EEGDataset(train_x,train_label)
test_dataset = EEGDataset(test_x,test_label)

train_loader = Data.DataLoader(train_dataset,
                               batch_size=32,
                               shuffle=True)

test_loader = Data.DataLoader(test_dataset,
                               batch_size=32,
                               shuffle=False)

best_model = torch.load('best_cross_net.pt')
trn_fea_list = []
tst_fea_list = []
trn_label_list = []
tst_label_list = []

best_model.eval()
test_loader_iter = iter(test_loader)

for trn_data, trn_label in train_loader:
    if test_loader_iter:
        try:
            tst_data, tst_label = next(test_loader_iter)
        except:
            test_loader_iter = iter(test_loader)
            tst_data, tst_label = next(test_loader_iter)

    trn_label = trn_label.type(torch.LongTensor).to(device)
    tst_label = tst_label.type(torch.LongTensor).to(device)

    trn_data = trn_data.float().to(device)
    tst_data = tst_data.float().to(device)

    trn_fea, tst_fea, _, _, _, _= best_model(trn_data, tst_data)

    trn_fea_list.append(trn_fea.cpu())
    tst_fea_list.append(tst_fea.cpu())
    trn_label_list.append(trn_label.cpu())
    tst_label_list.append(tst_label.cpu())

train_feature =[]
for i in range(len(trn_fea_list)):
    trn_fea = trn_fea_list[i]
    train_feature.append(trn_fea.detach().numpy())
train_feature = np.concatenate(train_feature)

train_label = []
for i in range(len(trn_label_list)):
    trn_lab = trn_label_list[i]
    train_label.append(trn_lab.detach().numpy())
train_label = np.concatenate(train_label).reshape(-1)

test_feature =[]
for i in range(len(tst_fea_list)):
    tst_fea = tst_fea_list[i]
    test_feature.append(tst_fea.detach().numpy())
test_feature = np.concatenate(test_feature)

test_label = []
for i in range(len(tst_label_list)):
    tst_lab = tst_label_list[i]
    test_label.append(tst_lab.detach().numpy())
test_label = np.concatenate(test_label).reshape(-1)

all_fea = np.concatenate((train_feature, test_feature), axis=0)
all_label = np.concatenate((train_label, test_label)).reshape(-1)

tsne = TSNE(n_components=2, random_state=2021)
Xtsne = tsne.fit_transform(all_fea)
s_label = train_label
t_label = test_label

s_tsne = Xtsne[0:len(s_label), :]
t_tsne = Xtsne[len(s_label):, :]

# data1 = s_tsne[:, 0]
# u = data1.mean()  # 计算均值
# std = data1.std()  # 计算标准差
# s_tsne = s_tsne[np.abs(data1 - u) <= 3 * std]
# s_label = s_label[np.abs(data1 - u) <= 3 * std]
# data1 = s_tsne[:, 1]
# u = data1.mean()  # 计算均值
# std = data1.std()  # 计算标准差
# s_tsne = s_tsne[np.abs(data1 - u) <= 3 * std]
# s_label = s_label[np.abs(data1 - u) <= 3 * std]
#
# data2 = t_tsne[:, 0]
# u = data2.mean()  # 计算均值
# std = data2.std()  # 计算标准差
# t_tsne = t_tsne[np.abs(data2 - u) <= 3 * std]
# t_label = t_label[np.abs(data2 - u) <= 3 * std]
# data1 = t_tsne[:, 1]
# u = data1.mean()  # 计算均值
# std = data1.std()  # 计算标准差
# t_tsne = t_tsne[np.abs(data1 - u) <= 3 * std]
# t_label = t_label[np.abs(data1 - u) <= 3 * std]

Xtsne = np.concatenate((s_tsne, t_tsne))

x_min, x_max = Xtsne.min(0), Xtsne.max(0)
Xtsne = (Xtsne - x_min) / (x_max - x_min)

n_class = len(np.unique(s_label))
color = ['red',  'blue']
labels = ["s", "t"]
# 显示图例
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18, }

plt.figure(figsize=(9,9))
plt.rcParams['savefig.dpi'] = 512
for i in range(len(Xtsne)):
    if i < len(s_label):
        c = int(all_label[i])
        plt.scatter(Xtsne[i, 0], Xtsne[i, 1], marker="x", alpha=0.5, c=color[c], s=30)
    else:
        c = int(all_label[i])
        plt.scatter(Xtsne[i, 0], Xtsne[i, 1], alpha=0.8, facecolors='none', edgecolors=color[c], s=30)

# 绘制各类均值中心
s_center = []
for c in range(n_class):
    center = np.mean((Xtsne[0: len(s_label), :])[s_label == c, :], axis=0)
    center = trim_mean((Xtsne[0: len(s_label), :])[s_label == c, :], 0.2, axis=0)
    s_center.append(center)
    plt.scatter(center[0], center[1], alpha=1, color=color[c], label=labels[c], marker="s", s=60)

t_center = []
for c in range(n_class):
    center = np.mean((Xtsne[len(s_label):, :])[t_label == c, :], axis=0)
    center = trim_mean((Xtsne[len(s_label):, :])[t_label == c, :], 0.2, axis=0)
    t_center.append(center)
    plt.scatter(center[0], center[1], alpha=1, color=color[c], marker="*", s=120)
plt.show()
