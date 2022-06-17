import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from bayes_opt import BayesianOptimization
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from evaluate_model import evaluate_model_performance_two_class
import warnings
import numpy as np
from torch.autograd import Function
warnings.filterwarnings("ignore")


class GradientReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class GCN(nn.Module):
    def __init__(self, adjacency, in_channel, out_channel, device):
        super(GCN, self).__init__()
        self.device = device
        self.adjacency = adjacency + torch.eye(adjacency.size(0)).to(device)
        self.D = torch.diag(torch.sum(self.adjacency, 1))
        self.D = self.D.inverse().sqrt()
        self.adjacency = torch.mm(torch.mm(self.D, self.adjacency), self.D)  # 对称归一化后的邻接矩阵
        in_channel = int(in_channel)
        out_channel = int(out_channel)
        self.w = nn.Parameter(torch.rand(in_channel, out_channel, requires_grad=True))
        # print(self.w.size())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.w)  # 返回两个数组的矩阵乘积
        hidden = torch.matmul(self.adjacency, support)
        return hidden

class Model(nn.Module):
    def __init__(self, adjacency, frequency_feature_size,
                 time_feature_size, demographic_size, hidden_1, output_size, channel, device):
        super(Model, self).__init__()
        self.device = device
        self.gcn1 = GCN(adjacency, int(frequency_feature_size), hidden_1, device).to(device)
        self.gcn2 = GCN(adjacency, hidden_1, hidden_1, device).to(device)
        self.gcn3 = GCN(adjacency, time_feature_size, hidden_1, device).to(device)
        self.gcn4 = GCN(adjacency, hidden_1, hidden_1, device).to(device)

        self.fc1 = nn.Linear(demographic_size, hidden_1)
        self.output_1 = nn.Linear((2 * channel+1) * hidden_1, output_size)
        # print(output_size)

        self.fc2 = nn.Linear((2 * channel + 1) * hidden_1, hidden_1)
        self.fc3 = nn.Linear(hidden_1, frequency_feature_size)
        self.fc4 = nn.Linear(hidden_1, time_feature_size)

        # 特征维度：(2 * channel + 1) * hidden_1
        self.dis = nn.Sequential(
            nn.Linear((2 * channel + 1) * hidden_1, 2),
            nn.Dropout(p=0.2),
            nn.LogSoftmax(0)
        )

    def forward(self, input_frequency, input_time_domain, demographics):
        # print(self.device)
        hidden_frequency = torch.relu(self.gcn1(input_frequency))
        hidden_frequency = F.dropout(hidden_frequency, p=0.6, training=self.training)
        hidden_frequency = self.gcn2(hidden_frequency)
        hidden_frequency = F.dropout(hidden_frequency, p=0.6, training=self.training)

        hidden_time = torch.relu(self.gcn3(input_time_domain))
        hidden_time = F.dropout(hidden_time, p=0.6, training=self.training)
        hidden_time = self.gcn4(hidden_time)
        hidden_time = F.dropout(hidden_time, p=0.6, training=self.training)

        hidden_demographic = torch.relu(self.fc1(demographics))
        hidden_demographic = torch.unsqueeze(hidden_demographic, dim=1)
        hidden_all = torch.cat((hidden_frequency, hidden_time, hidden_demographic), dim=1)
        hidden_all = hidden_all.reshape(hidden_all.size(0), -1)
        output = self.output_1(hidden_all)
        output = torch.squeeze(output, dim=-1)
        return hidden_all, output, torch.sigmoid(output)

    def generate(self, hidden_decode):
        hidden = torch.relu(self.fc2(hidden_decode))
        generate_frequency = self.fc3(hidden)
        generate_time_feature = self.fc4(hidden)
        return torch.cat((generate_frequency, generate_time_feature), dim=-1)

    def discriminator(self, hidden_feature, alpha):
        hidden_feature = hidden_feature.view(hidden_feature.shape[0], -1)
        # print(hidden_feature.size())
        if self.training:
            re_feature = GradientReverseLayer.apply(hidden_feature, alpha)
            dis_output = self.dis(re_feature)
        else:
            dis_output = self.dis(hidden_feature)
        return dis_output


def run_experiment(index, learning_rate, hidden_1):
    path = '../..'
    labels = np.zeros(shape=[0, 1])
    predicted_labels_prob = np.zeros(shape=[0])

    features_ = np.load(path + '/all_patient_frequency_features_15360.npy').reshape(-1, 16, 5)
    features_de = np.load(path + '/all_patient_de_features_15360.npy').reshape(-1, 16, 5)
    features_dasm = np.load(path+'/all_patient_dasm_features_15360.npy').reshape(-1, 8*5)
    features_rasm = np.load(path+'/all_patient_rasm_features_15360.npy').reshape(-1, 8*5)
    features_dcau = np.load(path+'/all_patient_dcau_features_15360.npy').reshape(-1, 6*5)
    features_ = np.concatenate((features_, features_de,
                                # features_dasm, features_rasm,
                                # features_dcau
                                ), axis=1)
    if index == -1:
        features_ = features_[:, :, :]
        frequency_feature_size = 160
        channel = 16
    else:
        features_ = features_[:, :, index]
        frequency_feature_size = 32
        channel = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_feature_size = 224
    demographic_size = 2 + 8*5*2 + 6*5
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
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 连接方式

    labels_ = np.load(path + '/all_patient_frequency_labels_15360.npy').reshape(-1, )
    age_and_sex_features = np.load(path + '/all_patient_age_and_sex.npy').reshape(-1, 2)
    age_and_sex_features = np.concatenate((age_and_sex_features, features_dasm, features_rasm, features_dcau), axis=-1)
    time_mean_and_std = np.load(path + '/all_patient_time_features.npy').reshape(-1, 16, 14)

    hc_index = 2
    scz_index = 3
    labels_hc_index = np.argwhere(labels_ == hc_index)
    features_hc = features_[labels_hc_index]
    age_and_sex_features_hc = age_and_sex_features[labels_hc_index]
    time_mean_and_std_hc = time_mean_and_std[labels_hc_index]
    features_hc = np.concatenate(
        (features_hc.reshape(len(features_hc), -1),
         time_mean_and_std_hc.reshape(len(time_mean_and_std_hc), -1),
         age_and_sex_features_hc.reshape(len(age_and_sex_features_hc), -1),
         ),
        axis=1)
    labels_hc = np.zeros(shape=[len(labels_hc_index)])

    labels_scz_index = np.argwhere(labels_ == scz_index)
    features_scz = features_[labels_scz_index]
    age_and_sex_features_scz = age_and_sex_features[labels_scz_index]
    time_mean_and_std_scz = time_mean_and_std[labels_scz_index]
    features_scz = np.concatenate((features_scz.reshape(len(features_scz), -1),
                                   time_mean_and_std_scz.reshape(len(time_mean_and_std_scz), -1),
                                   age_and_sex_features_scz.reshape(len(age_and_sex_features_scz), -1),
                                   ), axis=1)
    labels_scz = np.ones(shape=[len(labels_scz_index)])

    features_ = np.concatenate((features_hc, features_scz), axis=0)
    labels_ = np.concatenate((labels_hc, labels_scz), axis=0)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    count = 0
    loss_function = torch.nn.BCEWithLogitsLoss()
    for train_idx, test_idx in kf.split(features_, labels_):
        count += 1
        train_x, train_y, test_x, test_y = features_[train_idx], labels_[train_idx], \
                                           features_[test_idx], labels_[test_idx]

        train_x = train_x.reshape(len(train_x), -1)
        test_x = test_x.reshape(len(test_x), -1)
        labels = np.concatenate((labels, test_y.reshape(-1, 1)), axis=0)
        adjacency = torch.FloatTensor(np.array(adjacency)).to(device)
        model = Model(adjacency, frequency_feature_size // 16,
                      time_feature_size // 16, demographic_size, hidden_1, 1, channel, device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        model.train()
        min_max_scaler = preprocessing.StandardScaler()
        train_x = min_max_scaler.fit_transform(train_x)
        test_x = min_max_scaler.transform(test_x)
        train_frequency = train_x[:, :frequency_feature_size].reshape(-1, 16, frequency_feature_size // 16)
        train_time_features = train_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1,
                                                                                                                    16,
                                                                                                                    time_feature_size // 16)
        train_demographics = train_x[:, frequency_feature_size + time_feature_size:]

        for epoch in range(10):
            _, train_pred_prob, train_pre_prob_ = model(train_frequency, train_time_features, train_demographics)
            loss = loss_function(train_pred_prob, torch.FloatTensor(train_y).to(device))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            model.eval()
            test_frequency = test_x[:, :frequency_feature_size].reshape(-1, 16, frequency_feature_size // 16)
            test_time_features = test_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(
                -1,
                16,
                time_feature_size // 16)

            test_demographics = test_x[:, frequency_feature_size + time_feature_size:]
            _, predicted_y_prob, predicted_y_prob_ = model(test_frequency, test_time_features, test_demographics)
            test_loss = loss_function(predicted_y_prob, torch.FloatTensor(test_y).to(device))

            predicted_y_prob_ = predicted_y_prob_.detach().cpu().numpy()
            acc, recall, precision, f1, specificity, auc = evaluate_model_performance_two_class(test_y,
                                                                                                predicted_y_prob_,
                                                                                                flag=False)
            print("validation {:01d} | validation {:02d} | train_loss(s) {:.4f} | test_loss {:.4f} | Acc {:.4f} | "
                  "recall {:.4f} | precision {:.4f} | F1 {:.4f} | specificity {:.4f} | AUC {:.4f}"
                  .format(count, epoch, loss.detach().numpy(), test_loss, acc, recall, precision, f1, specificity, auc))

        predicted_labels_prob = np.concatenate((predicted_labels_prob, predicted_y_prob_), axis=0)
    acc, recall, precision, f1, specificity, auc = evaluate_model_performance_two_class(labels,
                                                                                        predicted_labels_prob,
                                                                                        flag=True)
    return acc, recall, precision, f1, specificity, auc


def optimize_parameters(learning_rate, hidden_1):
    learning_rate = 10 ** learning_rate
    hidden_1 = 2 ** int(hidden_1)
    print(f'learning_rate: {learning_rate}, hidden_size: {hidden_1}')
    all_acc = []
    all_recall = []
    all_precision = []
    all_f1 = []
    all_specificity = []
    all_auc = []
    for i in range(10):
        print(i, end=',')
        acc, recall, precision, f1, specificity, auc = run_experiment(-1, learning_rate, hidden_1)
        all_acc.append(acc)
        all_recall.append(recall)
        all_precision.append(precision)
        all_f1.append(f1)
        all_specificity.append(specificity)
        all_auc.append(auc)

    print('final_result: auc--{:.3f}±{:.3f}-acc--{:.3f}±{:.3f}--precision--{:.3f}±{:.3f}-'
          '-recall--{:.3f}±{:.3f}--f1--{:.3f}±{:.3f}-'
          '-specificity--{:.3f}±{:.3f}'.format(np.mean(all_auc),
                                               np.std(all_auc),
                                               np.mean(all_acc),
                                               np.std(all_acc),
                                               np.mean(all_precision),
                                               np.std(all_precision),
                                               np.mean(all_recall),
                                               np.std(all_recall),
                                               np.mean(all_f1),
                                               np.std(all_f1),
                                               np.mean(all_specificity),
                                               np.std(all_specificity)))
    return np.mean(all_auc)


if __name__ == '__main__':
    train_rnn_BO = BayesianOptimization(
        optimize_parameters, {
            'learning_rate': (-5, 0),
            'hidden_1': (4, 8),
        }
    )
    train_rnn_BO.maximize()
    print(train_rnn_BO.max)
