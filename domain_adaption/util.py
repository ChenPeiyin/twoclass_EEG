import numpy as np
import torch
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import logging
from torch.utils.data import Dataset


def save_logging(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置打印级别
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    # 设置屏幕打印的格式
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 设置log保存
    fh = logging.FileHandler(file_name, encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info('Start print log......')


def load_data(path, index, balance=False):
    features_ = np.load(path + '/all_patient_frequency_features_15360.npy').reshape(-1, 16, 5)
    features_de = np.load(path + '/all_patient_de_features_15360.npy').reshape(-1, 16, 5)
    features_dasm = np.load(path + '/all_patient_dasm_features_15360.npy').reshape(-1, 8, 5)
    features_rasm = np.load(path + '/all_patient_rasm_features_15360.npy').reshape(-1, 8, 5)
    features_dcau = np.load(path + '/all_patient_dcau_features_15360.npy').reshape(-1, 6, 5)
    features_ = np.concatenate((features_, features_de,
                                features_dasm, features_rasm,
                                features_dcau
                                ), axis=1)
    if index == -1:
        features_ = features_[:, :, :]
    else:
        features_ = features_[:, :, index]

    labels_ = np.load(path + '/all_patient_frequency_labels_15360.npy').reshape(-1, )
    age_and_sex_features = np.load(path + '/all_patient_age_and_sex.npy').reshape(-1, 2)
    time_mean_and_std = np.load(path + '/all_patient_time_features.npy').reshape(-1, 16, 14)

    hc_index = 2
    scz_index = 3
    labels_hc_index = np.argwhere(labels_ == hc_index)
    features_hc = features_[labels_hc_index]
    age_and_sex_features_hc = age_and_sex_features[labels_hc_index]
    time_mean_and_std_hc = time_mean_and_std[labels_hc_index]
    features_hc = np.concatenate(
        (features_hc.reshape(len(features_hc), -1),
         age_and_sex_features_hc.reshape(len(age_and_sex_features_hc), -1),
         time_mean_and_std_hc.reshape(len(time_mean_and_std_hc), -1)
         ),
        axis=1)
    labels_hc = np.zeros(shape=[len(labels_hc_index)])

    labels_scz_index = np.argwhere(labels_ == scz_index)
    features_scz = features_[labels_scz_index]
    age_and_sex_features_scz = age_and_sex_features[labels_scz_index]
    time_mean_and_std_scz = time_mean_and_std[labels_scz_index]
    features_scz = np.concatenate((features_scz.reshape(len(features_scz), -1),
                                   age_and_sex_features_scz.reshape(len(age_and_sex_features_scz), -1),
                                   time_mean_and_std_scz.reshape(len(time_mean_and_std_scz), -1)
                                   ), axis=1)
    labels_scz = np.ones(shape=[len(labels_scz_index)])

    features_ = np.concatenate((features_hc, features_scz), axis=0)
    labels_ = np.concatenate((labels_hc, labels_scz), axis=0)
    labels_ = labels_.astype(np.int64)
    if balance:
        features_, labels_ = SMOTE().fit_resample(features_, labels_)
    return shuffle(features_, labels_)


def shuffle(features, labels):
    index = np.arange(len(features))
    np.random.shuffle(index)
    features = features[index]
    labels = labels[index]
    return features, labels

class EEGDataset(Dataset):
    def __init__(self, X, y):
        X = X.astype('float')
        # y = y.astype('int')
        dataSize = X.shape
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.len = len(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
