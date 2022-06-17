import numpy as np

from GCN import Model
from util import *
import torch
from sklearn import preprocessing
import torch.nn.functional as F
import torch.nn as nn
import logging
from bayes_opt import BayesianOptimization


def pre_train_GCN(learning_rate, l2_regularization, hidden_size):
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization
    hidden_size = 2 ** int(hidden_size)

    print(f"learning_rate: {learning_rate},"
          f"l2_regularization: {l2_regularization},"
          f"hidden_size: {hidden_size}")
    path = 'E:/damo/data_extract_TUH/'

    train_psd_features = np.load(path+'data_extract/filter/train_psd_feature.npy', allow_pickle=True)
    train_de_features = np.load(path+'data_extract/filter/train_de_feature.npy', allow_pickle=True)
    train_time_features = np.load(path+'data_extract/filter/train_time_feature.npy', allow_pickle=True)
    train_age_and_sex = np.load(path+'data_extract/filter/train_age_and_sex.npy', allow_pickle=True)

    test_psd_features = np.load(path+'data_extract/filter/test_psd_feature.npy', allow_pickle=True)
    test_de_features = np.load(path+'data_extract/filter/test_de_feature.npy', allow_pickle=True)
    test_time_features = np.load(path+'data_extract/filter/test_time_feature.npy', allow_pickle=True)
    test_age_and_sex = np.load(path+'data_extract/filter/test_age_and_sex.npy', allow_pickle=True)

    time_feature_size = 224
    demographic_size = 2
    frequency_feature_size = 160

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    masked_index = np.random.randint(0, 16)  # masked channel
    train_labels = np.concatenate((train_psd_features[:, masked_index, :],
                                   train_de_features[:, masked_index, :],
                                   train_time_features[:, masked_index, :]), axis=-1)  # train_labels对通道掩码后的三个特征数据
    test_labels = np.concatenate((test_psd_features[:, masked_index, :],
                                  test_de_features[:, masked_index, :],
                                  test_time_features[:, masked_index, :]), axis=-1)

    train_psd_features[:, masked_index, :] *= 0  # train_psd_features[:, masked_index, :] = train_psd_features[:, masked_index, :] * 0 使掩码的该通道的数据为0
    train_de_features[:, masked_index, :] *= 0
    train_time_features[:, masked_index, :] *= 0

    test_psd_features[:, masked_index, :] *= 0
    test_de_features[:, masked_index, :] *= 0
    test_time_features[:, masked_index, :] *= 0

    train_x = np.concatenate((train_psd_features.reshape(len(train_psd_features), -1),
                              train_de_features.reshape(len(train_de_features), -1),
                              train_time_features.reshape(len(train_time_features), -1),
                              train_age_and_sex.reshape(len(train_age_and_sex), -1)), axis=-1)  # train_x是掩码某个通道后4个特征的数据

    test_x = np.concatenate((test_psd_features.reshape(len(test_psd_features), -1),
                             test_de_features.reshape(len(test_de_features), -1),
                             test_time_features.reshape(len(test_time_features), -1),
                             test_age_and_sex.reshape(len(test_age_and_sex), -1)), axis=-1)

    model = Model(adjacency, frequency_feature_size // 16,
                  time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    loss_function = nn.L1Loss(reduction='mean')

    for epoch in range(30):
        min_max_scaler = preprocessing.MinMaxScaler()
        train_x = min_max_scaler.fit_transform(train_x)  # 对特征数据归一化
        test_x = min_max_scaler.fit_transform(test_x)

        train_hidden, _, _ = model(
            torch.FloatTensor(train_x[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
            torch.FloatTensor(
                train_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16, 14)).to(
                device),
            torch.FloatTensor(train_x[:, frequency_feature_size + time_feature_size:]).to(device))
        generated_x_train = model.generate(train_hidden)
        print(train_hidden.size())


        loss = F.mse_loss(generated_x_train, torch.FloatTensor(train_labels).to(device))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        test_hidden, _, _ = model(torch.FloatTensor(test_x[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
                                  torch.FloatTensor(
                                      test_x[:,
                                      frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16,
                                                                                                                 14)).to(
                                      device),
                                  torch.FloatTensor(test_x[:, frequency_feature_size + time_feature_size:]).to(device))
        generated_x_test = model.generate(test_hidden)

        mse_test = F.mse_loss(generated_x_test, torch.FloatTensor(test_labels).to(device), reduction='mean')
        mae_test = loss_function(generated_x_test, torch.FloatTensor(test_labels).to(device))

        logging.info("epoch {:02d} | mse_train {:.4f} | mse_test {:.4f} | mae_test {:.4f}"
                     .format(epoch, loss, mse_test, mae_test))

    return -mse_test.detach().cpu().numpy(), model


if __name__ == '__main__':
    # save_logging('pre_train_mask_generation_TUH.log')
    reward_train = BayesianOptimization(
        pre_train_GCN, {
            'hidden_size': (5, 8),
            'learning_rate': (-7, -2),
            'l2_regularization': (-6, -1),

        }
    )
    reward_train.maximize(n_iter=1000)
    logging.info(reward_train.max)


