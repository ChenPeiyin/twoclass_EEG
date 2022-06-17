import numpy as np
import os
import os.path as osp
import datetime
from GCN import Model, GradientReverseLayer
from util import *
import torch
from sklearn import preprocessing
from scipy.io import savemat
import torch.nn.functional as F
import torch.nn as nn
import logging
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import normalize


def pre_train_GCN(learning_rate, l2_regularization, hidden_size, len_src, len_tar, alpha, beta):
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization
    hidden_size = 2 ** int(hidden_size)
    len_src = len_src
    len_tar = len_tar
    alpha = alpha  # TUH vs. src
    beta = beta  # TUH vs. tar

    mode = 'tuh&tar'
    normalization = False

    print(f"learning_rate: {learning_rate},"
          f"l2_regularization: {l2_regularization},"
          f"hidden_size: {hidden_size}",
          f"alpha: {alpha}",
          f"beta: {beta}")

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

    tuh_path = '../data_extract_TUH/data_extract/filter'
    tuh_data1 = []
    tuh_data2 = []
    for folder in os.listdir(tuh_path):
        print(folder)
        domain = os.path.abspath(tuh_path)
        folderpath = os.path.join(domain, folder)
        print(folderpath)
        if folder.find('test') != -1:
            if folder.find('sex') != -1:
                raw_data = np.load(folderpath, allow_pickle=True)
                print(len(raw_data))
                tuh_data1.append(np.squeeze(raw_data))
            else:
                raw_data = np.load(folderpath, allow_pickle=True)
                if normalization is True:
                    raw_data = normalize(raw_data.reshape(-1, raw_data.shape[-1]), norm='l2')
                    raw_data = raw_data.reshape(-1, 16, raw_data.shape[-1])
                else:
                    raw_data = raw_data
                tuh_data2.append(np.squeeze(raw_data))

    tuh_labels = np.concatenate((tuh_data2[0][:, masked_index, :],
                                 tuh_data2[1][:, masked_index, :],
                                 tuh_data2[2][:, masked_index, :]), axis=-1)

    tuh_data2[0][:, masked_index, :] *= 0
    tuh_data2[1][:, masked_index, :] *= 0
    tuh_data2[2][:, masked_index, :] *= 0

    tuh_x = np.concatenate((tuh_data1[0].reshape(len(tuh_data1[0]), -1),
                            tuh_data2[0].reshape(len(tuh_data2[0]), -1),
                            tuh_data2[1].reshape(len(tuh_data2[1]), -1),
                            tuh_data2[2].reshape(len(tuh_data2[2]), -1)), axis=-1)

    src_path = "../data_sample/data_sample/source"
    src_data1 = []
    src_data2 = []
    src_label = []
    for folder in os.listdir(src_path):
        print(folder)
        domain = os.path.abspath(src_path)
        folderpath = os.path.join(domain, folder)
        print(folderpath)
        if folder.find('sex') != -1:
            raw_data = np.load(folderpath, allow_pickle=True)
            print(len(raw_data))
            src_data1.append(np.squeeze(raw_data))
        elif folder.find('label') != -1:
            raw_label = np.load(folderpath, allow_pickle=True)
            print(len(raw_label))
            src_label.append(np.squeeze(raw_label))
        else:
            raw_data = np.load(folderpath, allow_pickle=True)
            if normalization is True:
                raw_data = normalize(raw_data.reshape(-1, raw_data.shape[-1]), norm='l2')
                raw_data = raw_data.reshape(-1, 16, raw_data.shape[-1])
            else:
                raw_data = raw_data
            src_data2.append(np.squeeze(raw_data))

    src_labels = np.concatenate((src_data2[0][:, masked_index, :],
                              src_data2[1][:, masked_index, :],
                              src_data2[2][:, masked_index, :]), axis=-1)

    src_data2[0][:, masked_index, :] *= 0
    src_data2[1][:, masked_index, :] *= 0
    src_data2[2][:, masked_index, :] *= 0

    src_x = np.concatenate((src_data2[0].reshape(len(src_data2[0]), -1),
                              src_data2[1].reshape(len(src_data2[1]), -1),
                              src_data2[2].reshape(len(src_data2[2]), -1),
                              src_data1[0].reshape(len(src_data1[0]), -1)), axis=-1)  # (188, 386)

    tar_path = "../data_sample/data_sample/target"
    tar_data1 = []
    tar_data2 = []
    tar_label = []
    for folder in os.listdir(tar_path):
        print(folder)
        domain = os.path.abspath(tar_path)
        folderpath = os.path.join(domain, folder)
        print(folderpath)
        if folder.find('sex') != -1:
            raw_data = np.load(folderpath, allow_pickle=True)
            print(len(raw_data))
            tar_data1.append(np.squeeze(raw_data))
        elif folder.find('label') != -1:
            raw_label = np.load(folderpath, allow_pickle=True)
            print(len(raw_label))
            tar_label.append(np.squeeze(raw_label))
        else:
            raw_data = np.load(folderpath, allow_pickle=True)
            if normalization is True:
                raw_data = normalize(raw_data.reshape(-1, raw_data.shape[-1]), norm='l2')
                raw_data = raw_data.reshape(-1, 16, raw_data.shape[-1])
            else:
                raw_data = raw_data
            tar_data2.append(np.squeeze(raw_data))

    tar_labels = np.concatenate((tar_data2[0][:, masked_index, :],
                                 tar_data2[1][:, masked_index, :],
                                 tar_data2[2][:, masked_index, :]), axis=-1)

    tar_data2[0][:, masked_index, :] *= 0
    tar_data2[1][:, masked_index, :] *= 0
    tar_data2[2][:, masked_index, :] *= 0

    tar_x = np.concatenate((tar_data2[0].reshape(len(tar_data2[0]), -1),
                            tar_data2[1].reshape(len(tar_data2[1]), -1),
                            tar_data2[2].reshape(len(tar_data2[2]), -1),
                            tar_data1[0].reshape(len(tar_data1[0]), -1)), axis=-1)  # (250, 386)

    if mode == "tuh&src":
        train_x = tuh_x[:int(len_src * src_x.shape[0]), :]
        test_x = src_x[:int(len_src * src_x.shape[0]), :]
        train_labels = tuh_labels[:int(len_src * src_x.shape[0]), :]
        test_labels = src_labels[:int(len_src * src_x.shape[0]), :]
    elif mode == "tuh&tar":
        train_x = tuh_x[:int(len_tar * tar_x.shape[0]), :]
        test_x = tar_x[:int(len_tar * tar_x.shape[0]), :]
        train_labels = tuh_labels[:int(len_tar * tar_x.shape[0]), :]
        test_labels = tar_labels[:int(len_tar * tar_x.shape[0]), :]
    else:
        train_x = tuh_x[:int(len_tar * tar_x.shape[0]), :]
        test_x1 = src_x[:int(len_src * src_x.shape[0]), :]
        test_x2 = tar_x[:int(len_tar * tar_x.shape[0]), :]
        test_x = np.concatenate((test_x1, test_x2))
        train_labels = tuh_labels[:int(len_tar * tar_x.shape[0]), :]
        test_labels1 = src_labels[:int(len_src * src_x.shape[0]), :]
        test_labels2 = tar_labels[:int(len_tar * tar_x.shape[0]), :]
        test_labels = np.concatenate((test_labels1, test_labels2))

    model = Model(adjacency, frequency_feature_size // 16,
                  time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    loss_function = nn.L1Loss(reduction='mean')
    dis_criterion = F.nll_loss

    for epoch in range(100):
        model.train()

        min_max_scaler = preprocessing.MinMaxScaler()
        train_x = min_max_scaler.fit_transform(train_x)
        test_x = min_max_scaler.fit_transform(test_x)

        train_hidden, _, _ = model(
            torch.FloatTensor(train_x[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
            torch.FloatTensor(
                train_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16, 14)).to(
                device),
            torch.FloatTensor(train_x[:, frequency_feature_size + time_feature_size:]).to(device))
        # print(train_hidden.size())
        generated_x_train = model.generate(train_hidden)
        discriminate_x_train = model.discriminator(train_hidden, alpha=1)

        pre_loss = F.mse_loss(generated_x_train, torch.FloatTensor(train_labels).to(device))

        if mode == 'tuh&src':
            test_hidden, _, _ = model(
                torch.FloatTensor(test_x[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
                torch.FloatTensor(
                    test_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16, 14)).to(
                    device),
                torch.FloatTensor(test_x[:, frequency_feature_size + time_feature_size:]).to(device))
            generated_x_test = model.generate(test_hidden)
            discriminate_x_test = model.discriminator(test_hidden, alpha=1)

            train_bs = len(train_x)
            source_label = torch.ones(train_bs)
            source_label = source_label.long()

            test_bs = len(test_x)
            target_label = torch.zeros(test_bs)
            target_label = target_label.long()

            domain_output = torch.cat((discriminate_x_train, discriminate_x_test), dim=0)
            domain_label = torch.cat((source_label, target_label), dim=0)

            if torch.cuda.is_available():
                domain_output = domain_output.cuda()
                domain_label = domain_label.cuda()

            dis_loss = dis_criterion(domain_output, domain_label)

        elif mode == 'tuh&tar':
            test_hidden, _, _ = model(
                torch.FloatTensor(test_x[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
                torch.FloatTensor(
                    test_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16, 14)).to(
                    device),
                torch.FloatTensor(test_x[:, frequency_feature_size + time_feature_size:]).to(device))
            generated_x_test = model.generate(test_hidden)
            discriminate_x_test = model.discriminator(test_hidden, alpha=1)

            train_bs = len(train_x)
            source_label = torch.ones(train_bs)
            source_label = source_label.long()

            test_bs = len(test_x)
            target_label = torch.zeros(test_bs)
            target_label = target_label.long()

            domain_output = torch.cat((discriminate_x_train, discriminate_x_test), dim=0)
            domain_label = torch.cat((source_label, target_label), dim=0)

            if torch.cuda.is_available():
                domain_output = domain_output.cuda()
                domain_label = domain_label.cuda()

            dis_loss = dis_criterion(domain_output, domain_label)

        # 两两对抗
        else:
            test_hidden1, _, _ = model(
                torch.FloatTensor(test_x1[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
                torch.FloatTensor(
                    test_x1[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16,
                                                                                                         14)).to(
                    device),
                torch.FloatTensor(test_x1[:, frequency_feature_size + time_feature_size:]).to(device))
            generated_x_test1 = model.generate(test_hidden1)
            discriminate_x_test1 = model.discriminator(test_hidden1, alpha=1)

            test_hidden2, _, _ = model(
                torch.FloatTensor(test_x2[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
                torch.FloatTensor(
                    test_x2[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16,
                                                                                                          14)).to(
                    device),
                torch.FloatTensor(test_x2[:, frequency_feature_size + time_feature_size:]).to(device))
            generated_x_test2 = model.generate(test_hidden2)
            discriminate_x_test2 = model.discriminator(test_hidden2, alpha=1)

            train_bs = len(train_x)
            source_label = torch.ones(train_bs)
            source_label = source_label.long()

            test_bs1 = len(test_x1)
            target_label1 = torch.zeros(test_bs1)
            target_label1 = target_label1.long()

            test_bs2 = len(test_x2)
            target_label2 = torch.zeros(test_bs2)
            target_label2 = target_label2.long()

            domain_output1 = torch.cat((discriminate_x_train, discriminate_x_test1), dim=0)
            domain_label1 = torch.cat((source_label, target_label1), dim=0)
            domain_output2 = torch.cat((discriminate_x_train, discriminate_x_test2), dim=0)
            domain_label2 = torch.cat((source_label, target_label2), dim=0)

            if torch.cuda.is_available():
                domain_output1 = domain_output1.cuda()
                domain_label1 = domain_label1.cuda()
                domain_output2 = domain_output2.cuda()
                domain_label2 = domain_label2.cuda()

            dis_loss1 = dis_criterion(domain_output1, domain_label1)
            dis_loss2 = dis_criterion(domain_output2, domain_label2)
            dis_loss = alpha * dis_loss1 + beta * dis_loss2

        if mode == "tuh&src":
            loss = pre_loss + alpha * dis_loss
        elif mode == "tuh&tar":
            loss = pre_loss + beta * dis_loss
        else:
            loss = pre_loss + dis_loss

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

        logging.info("epoch {:02d} | mse_train {:.4f} | da_train {:.4f} | loss_train {:.4f} | mse_test {:.4f} | mae_test {:.4f}"
                     .format(epoch, pre_loss, dis_loss, loss, mse_test, mae_test))

        if mode == "tuh&src":
            checkpoint_path = 'best_ts_net2.pth.tar'
        elif mode == "tuh&tar":
            checkpoint_path = 'best_tt_net2.pth.tar'
        else:
            checkpoint_path = 'best_tst_net2.pth.tar'
        torch.save(model.state_dict(), checkpoint_path)

    return -mse_test.detach().cpu().numpy()

if __name__ == '__main__':
    # save_logging('pre_train_mask_generation_TUH.log')
    log_file = '%s.txt' % datetime.date.today()
    working_dir = osp.dirname(osp.abspath(__file__))

    logs_dir = osp.join(working_dir, 'logs')
    if not osp.isdir(logs_dir):
        os.makedirs(logs_dir)

    reward_train = BayesianOptimization(
        pre_train_GCN, {
            'hidden_size': (5, 8),
            'learning_rate': (-7, -2),
            'l2_regularization': (-6, -1),
            'len_src': (0.3, 0.5),
            'len_tar': (0.3, 0.5),
            'alpha': (0.5, 1),
            'beta': (0.5, 1)
        }
    )
    reward_train.maximize(n_iter=100)
    logging.info(reward_train.max)

    out = pre_train_GCN(reward_train.max['params']['learning_rate'], reward_train.max['params']['l2_regularization'], reward_train.max['params']['hidden_size'],
                        reward_train.max['params']['len_src'], reward_train.max['params']['len_tar'],
                        reward_train.max['params']['alpha'], reward_train.max['params']['beta'])

    file_name = "mix_pretrain_model.mat"
    savemat(file_name,{'learning_rate':reward_train.max['params']['learning_rate'],
                       'l2_regularization':reward_train.max['params']['l2_regularization'],
                       'hidden_size':reward_train.max['params']['hidden_size'],
                       'len_src':reward_train.max['params']['len_src'],
                       'len_tar':reward_train.max['params']['len_tar'],
                        'alpha':reward_train.max['params']['alpha'],
                       'beta':reward_train.max['params']['beta']
    })




