import numpy as np
import torch
from GCN import Model, GradientReverseLayer
from util import EEGDataset
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn import preprocessing
import pandas as pd
from braindecode.torch_ext.util import set_random_seeds
from torch.backends import cudnn
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
from bayes_opt import BayesianOptimization
import logging
import random


gamma = 0.5  # Discriminator loss weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epoches = 500
loss_function = F.nll_loss

item = 10
learning_rate = 10 ** -2
l2_regularization = 10 ** -3
hidden_size = 2 ** int(5)
len_src = 0.3
len_tar = 0.3  # The optimal parameters obtained by pre-training

pre_train = True
DA = False
Augm = False
normalization = True
file = 'tuh&src&tar'

adjacency = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
adjacency = torch.FloatTensor(np.array(adjacency)).to(device)
time_feature_size = 224
demographic_size = 2
frequency_feature_size = 160

if pre_train is True:
    if file == 'tuh&src&tar':
        checkpoint_path = "best_tst_net1.pth.tar"
    elif file == "tuh&src":
        checkpoint_path = 'best_ts_net1.pth.tar'
    else:
        checkpoint_path = 'best_st_net1.pth.tar'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    pre_train_net = Model(adjacency, frequency_feature_size // 16,
                  time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)
    pre_train_net.load_state_dict(checkpoint)
    hidden_size = pre_train_net.dis[0].in_features
else:
    pre_train_net = Model(adjacency, frequency_feature_size // 16,
                  time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)

fea_dim = (2 * 16 + 1) * hidden_size

path = "../data_sample/data_sample/"

if normalization is True:
    train_psd_features = np.load(path + 'source/patient_psd.npy', allow_pickle=True)
    train_psd_features = normalize(train_psd_features.reshape(-1, train_psd_features.shape[-1]), norm='l2')
    train_psd_features = train_psd_features.reshape(-1, 16, train_psd_features.shape[-1])

    train_de_features = np.load(path + 'source/patient_de.npy', allow_pickle=True)
    train_de_features = normalize(train_de_features.reshape(-1, train_de_features.shape[-1]), norm='l2')
    train_de_features = train_de_features.reshape(-1, 16, train_de_features.shape[-1])

    train_time_features = np.load(path + 'source/patient_time.npy', allow_pickle=True)
    train_time_features = normalize(train_time_features.reshape(-1, train_time_features.shape[-1]), norm='l2')
    train_time_features = train_time_features.reshape(-1, 16, train_time_features.shape[-1])

    train_age_and_sex = np.load(path + 'source/patient_age_and_sex.npy', allow_pickle=True)

    test_psd_features = np.load(path + 'target/patient_psd.npy', allow_pickle=True)
    test_psd_features = normalize(test_psd_features.reshape(-1, test_psd_features.shape[-1]), norm='l2')
    test_psd_features = test_psd_features.reshape(-1, 16, test_psd_features.shape[-1])

    test_de_features = np.load(path + 'target/patient_de.npy', allow_pickle=True)
    test_de_features = normalize(test_de_features.reshape(-1, test_de_features.shape[-1]), norm='l2')
    test_de_features = test_de_features.reshape(-1, 16, test_de_features.shape[-1])

    test_time_features = np.load(path + 'target/patient_time.npy', allow_pickle=True)
    test_time_features = normalize(test_time_features.reshape(-1, test_time_features.shape[-1]), norm='l2')
    test_time_features = test_time_features.reshape(-1, 16, test_time_features.shape[-1])

    test_age_and_sex = np.load(path + 'target/patient_age_and_sex.npy', allow_pickle=True)

else:
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

if Augm is True:
    test_data1 = test_x[test_label == 0]
    test_label1 = test_label[test_label == 0]
    test_data2 = test_x[test_label == 1]
    test_label2 = test_label[test_label == 1]
    test_data3 = np.concatenate((test_data2, test_data2))
    test_data2 = np.concatenate((test_data3, test_data2))
    test_label3 = np.concatenate((test_label2, test_label2))
    test_label2 = np.concatenate((test_label3, test_label2))

    test_x = np.concatenate((test_data1, test_data2))
    test_label = np.concatenate((test_label1, test_label2))
    print(pd.value_counts(test_label))
else:
    pass

trn_data = train_x
trn_label = train_label
tst_data = test_x
tst_label = test_label
print(pd.value_counts(trn_label))
print(pd.value_counts(tst_label))

index = [i for i in range(len(trn_data))]
random.shuffle(index)
trn_data = trn_data[index]
trn_label = trn_label[index]

index = [i for i in range(len(tst_data))]
random.shuffle(index)
tst_data = tst_data[index]
tst_label = tst_label[index]

trn_len = int(trn_data.shape[0] * len_src)
tst_len = int(tst_data.shape[0] * len_tar)

trn_data = trn_data[trn_len:, :]
trn_label = trn_label[trn_len:]
tst_data = tst_data[tst_len:, :]
tst_label = tst_label[tst_len:]

class GradientReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, theta):
        ctx.theta = theta

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.theta

        return output, None

class DANet(nn.Module):
    def __init__(self, backbone, frequency_feature_size, time_feature_size, hidden_size, n_class=2, ):
        super(DANet, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.frequency_feature_size = frequency_feature_size
        self.time_feature_size = time_feature_size

        self.Fea = backbone

        self.Classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size,2),  # hidden_size
            nn.LogSoftmax(1)
        )

        self.Discrinminator = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size,2),  # hidden_size
            nn.LogSoftmax(1)
        )

    def forward(self, source_data, target_data, theta=1):
        # print(source_data.size(), target_data.size())
        s_fea,_,_ = self.Fea(source_data[:, :self.frequency_feature_size].reshape(-1, 16, 10),
                         source_data[:, self.frequency_feature_size:self.frequency_feature_size + time_feature_size].reshape(-1, 16, 14),
                         source_data[:, self.frequency_feature_size + self.time_feature_size:])
        # print(s_fea.size())

        s_clf = self.Classifier(s_fea)
        source_feature = s_fea.view(s_fea.shape[0], -1)
        # print(source_feature.size())

        t_fea,_,_ = self.Fea(target_data[:, :self.frequency_feature_size].reshape(-1, 16, 10),
                         target_data[:, self.frequency_feature_size:self.frequency_feature_size + time_feature_size].reshape(-1, 16, 14),
                         target_data[:, self.frequency_feature_size + self.time_feature_size:])
        # print(t_fea.size())

        t_clf = self.Classifier(t_fea)
        target_feature = t_fea.view(t_fea.shape[0], -1)

        if self.training:
            re_source_feature = GradientReverseLayer.apply(source_feature, theta)
            re_target_feature = GradientReverseLayer.apply(target_feature, theta)

            s_output = self.Discrinminator(re_source_feature)
            t_output = self.Discrinminator(re_target_feature)

        else:
            s_output = self.Discrinminator(s_fea)
            t_output = self.Discrinminator(t_fea)

        return source_feature, target_feature, s_clf, t_clf, s_output, t_output
        # return s_fea

trn_acc_list = []
trn_f1_list = []
trn_auc_list = []

tst_auc_list = []
tst_acc_list = []
tst_precision_list = []
tst_recall_list = []
tst_f1_list = []
tst_specificity_list = []

for _item in range(item):
    train_dataset = EEGDataset(trn_data,trn_label)
    test_dataset = EEGDataset(test_x,test_label)

    train_loader = Data.DataLoader(train_dataset,
                                   batch_size=32,
                                   shuffle=True)

    test_loader = Data.DataLoader(test_dataset,
                                   batch_size=32,
                                   shuffle=False)

    model = DANet(pre_train_net, frequency_feature_size, time_feature_size, hidden_size,n_class=2).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.Fea.parameters(), 'lr': learning_rate, "weight_decay": l2_regularization},
        {'params': model.Classifier.parameters(), 'lr': learning_rate, "weight_decay": l2_regularization},
        {'params': model.Discrinminator.parameters(), 'lr': learning_rate, "weight_decay": l2_regularization},
    ])

    best_trn_acc = 0
    best_tst_acc = 0

    for epoch in range(num_epoches):
        all_trn_pred = []
        all_trn_true = []
        all_tst_pred = []
        all_tst_true = []

        """ Training"""
        model.train()

        trn_correct = 0
        test_loader_iter = iter(test_loader)

        for source_data, source_label in train_loader:
            if test_loader_iter:
                try:
                    target_data, target_label = next(test_loader_iter)
                except:
                    test_loader_iter = iter(test_loader)
                    target_data, target_label = next(test_loader_iter)

            source_label = source_label.type(torch.LongTensor)
            target_label = target_label.type(torch.LongTensor)

            source_data = source_data.float().to(device)
            source_label = source_label.to(device)
            target_data = target_data.float().to(device)
            target_label = target_label.to(device)

            optimizer.zero_grad()

            source_feature, target_feature, source_clf, target_clf, source_dis, target_dis = model(source_data, target_data)

            clf_loss = loss_function(source_clf, source_label)

            s_pred = source_clf.argmax(axis=1)

            trn_total = len(train_loader.dataset)
            trn_correct += (s_pred == source_label.long()).sum().item()

            all_trn_pred.append(s_pred.detach().cpu().numpy())
            all_trn_true.append(source_label.detach().cpu().numpy())

            source_batch_size = len(source_data)
            source_label = torch.ones(source_batch_size)
            source_label = source_label.long()

            target_batch_size = len(target_data)
            target_label = torch.zeros(target_batch_size)
            target_label = target_label.long()

            domain_output = torch.cat((source_dis, target_dis), axis=0)
            domain_label = torch.cat((source_label, target_label), axis=0)

            domain_output = domain_output.to(device)
            domain_label = domain_label.to(device)

            dis_loss = gamma * loss_function(domain_output, domain_label)

            if DA is True:
                loss = clf_loss.cuda() + dis_loss.cuda()
            else:
                loss = clf_loss.cuda()

            loss.backward()
            optimizer.step()

        trn_acc = 100 * trn_correct / trn_total
        all_trn_pred = np.concatenate(all_trn_pred).reshape(-1)
        all_trn_true = np.concatenate(all_trn_true).reshape(-1)

        if trn_acc > best_trn_acc:
            best_trn_acc = trn_acc
            best_trn_epoch = epoch
            best_trn_f1 = f1_score(all_trn_pred, all_trn_true)
            best_trn_auc = roc_auc_score(all_trn_true, all_trn_pred)

        print("Train current epoch: {}, trn_clf_loss: {:.5f}, trn_dis_loss: {:.5f}, trn_total_loss: {:.5f}, trn_clf_acc: {}/{} ({:.2f}%)\n".
              format(epoch + 1, clf_loss, dis_loss, loss, trn_correct, trn_total, trn_acc))
        print("Best train accuracy / F1-score / AUC at epoch {} : {:.2f}% / {:.5f} / {:.5f}".format(best_trn_epoch+1, best_trn_acc, best_trn_f1, best_trn_auc))

        """ Testing"""
        model.eval()
        tst_correct = 0

        for i, (target_data, target_label) in enumerate(test_loader):
            target_data = target_data.float().to(device)
            target_label = target_label.to(device)

            tst_feature0, tst_feature, tst_clf0, tst_clf, tst_dis0, tst_dis = model(target_data, target_data)

            tst_pred = tst_clf.argmax(axis=1)
            tst_total = len(test_loader.dataset)
            tst_correct += (tst_pred == target_label.long()).sum().item()
            tst_clf_loss = F.nll_loss(tst_clf, target_label.long())

            all_tst_pred.append(tst_pred.detach().cpu().numpy())
            all_tst_true.append(target_label.detach().cpu().numpy())

        tst_acc = 100 * tst_correct / tst_total
        all_tst_pred = np.concatenate(all_tst_pred).reshape(-1)
        all_tst_true = np.concatenate(all_tst_true).reshape(-1)
        print('\nTest set: tst_clf_loss: {:.4f}, tst_clf_accuracy: {}/{} ({:.2f}%)\n'.format(tst_clf_loss, tst_correct,
                                                                                             tst_total, tst_acc))
        print(pd.value_counts(all_tst_pred))
        print(pd.value_counts(all_tst_true))

        if tst_acc > best_tst_acc:
            best_tst_epoch = epoch
            best_tst_auc = roc_auc_score(all_tst_true, all_tst_pred)
            best_tst_acc = tst_acc
            best_tst_precision = precision_score(all_tst_true, all_tst_pred)
            best_tst_recall = recall_score(all_tst_true, all_tst_pred)
            best_tst_f1 = f1_score(all_tst_pred, all_tst_true)
            cm = confusion_matrix(all_tst_true, all_tst_pred)
            tn, fp, fn, tp = confusion_matrix(all_tst_true, all_tst_pred).ravel()
            best_tst_specificity = tn / (tn + fp)

            # torch.save(model, 'best_cross_net.pt')
            checkpoint_path = 'best_cross_net.pth.tar'
            torch.save(model.state_dict(), checkpoint_path)
        print('Best auc--{:.5f}--acc--{:.5f}--precision--{:.5f}--recall--{:.5f}\n'
              'f1--{:.2f}--specificity--{:.5f} at epoch:{}'.format(best_tst_auc,
                                                            best_tst_acc,
                                                            best_tst_precision,
                                                            best_tst_recall,
                                                            best_tst_f1,
                                                            best_tst_specificity, best_tst_epoch))
        print("****************************************************************************")

    trn_auc_list.append(best_trn_auc)
    trn_acc_list.append(best_trn_acc)
    trn_f1_list.append(best_trn_f1)

    tst_auc_list.append(best_tst_auc)
    tst_acc_list.append(best_tst_acc)
    tst_precision_list.append(best_tst_precision)
    tst_recall_list.append(best_tst_recall)
    tst_f1_list.append(best_tst_f1)
    tst_specificity_list.append(best_tst_specificity)

mean_trn_auc = np.mean(np.array(trn_auc_list))
mean_trn_acc = np.mean(np.array(trn_acc_list))
mean_trn_f1 = np.mean(np.array(trn_f1_list))
print("Mean train AUC: {:.5f}\n"
      "Mean train ACC: {:.5f}%\n"
      "Mean train f1-score: {:.2f}".format(mean_trn_auc, mean_trn_acc, mean_trn_f1))

mean_tst_auc = np.mean(np.array(tst_auc_list))
mean_tst_acc = np.mean(np.array(tst_acc_list))
mean_tst_precision = np.mean(np.array(tst_precision_list))
mean_tst_recall = np.mean(np.array(tst_recall_list))
mean_tst_f1 = np.mean(np.array(tst_f1_list))
mean_tst_specificity = np.mean(np.array(tst_specificity_list))
print("Mean test AUC: {:.5f}\n"
      "Mean test ACC: {:.5f}%\n"
      "Mean test precision: {:.5f}\n"
      "Mean test recall: {:.5f}\n"
      "Mean test f1: {:.2f}\n"
      "Mean test specificity: {:.5f}".format(mean_tst_auc,
                                              mean_tst_acc,
                                              mean_tst_precision,
                                              mean_tst_recall,
                                              mean_tst_f1,
                                              mean_tst_specificity))








