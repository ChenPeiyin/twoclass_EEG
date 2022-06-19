import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.viz import plot_montage, plot_topomap

# label 1:Normal  label 3:Schizophrenia
mode = 'PSD'
path = "../data_sample/data_sample/"
for fre in range(5):

    if mode == 'PSD':
        src_features = np.squeeze(np.load(path+'source/patient_psd.npy', allow_pickle=True))  # (188, 16, 5)
        tar_features = np.squeeze(np.load(path + 'target/patient_psd.npy', allow_pickle=True))  # (250, 16, 5)
    else:
        src_features = np.squeeze(np.load(path+'source/patient_de.npy', allow_pickle=True))
        tar_features = np.squeeze(np.load(path + 'target/patient_de.npy', allow_pickle=True))

    src_label = np.load(path + 'source/patient_label.npy', allow_pickle=True).reshape(-1)
    tar_label = np.load(path + 'target/patient_label.npy', allow_pickle=True).reshape(-1)
    # 分成两类
    src_nor = src_features[src_label == 1]
    src_sch = src_features[src_label == 3]
    tar_nor = tar_features[tar_label == 1]
    tar_sch = tar_features[tar_label == 3]

    # 创建一个info结构
    ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4',
                     'T5', 'T6']
    ch_sfreq = 250
    info = mne.create_info(ch_names=ch_names, sfreq=ch_sfreq, ch_types='eeg')

    def evoked_plot(data, comment):
        evoked_data = data[:, np.newaxis]
        evoked_comment = comment
        evoked_array = mne.EvokedArray(evoked_data, info, comment=evoked_comment, nave=16)

        return evoked_array

    mean_src_nor = np.mean(src_nor, axis=0)
    mean_src_sch = np.mean(src_sch, axis=0)
    mean_tar_nor = np.mean(tar_nor, axis=0)
    mean_tar_sch = np.mean(tar_sch, axis=0)

    # 五个频段求平均
    a_evoked = np.mean(mean_src_nor, axis=-1)
    b_evoked = np.mean(mean_src_sch, axis=-1)
    c_evoked = np.mean(mean_tar_nor, axis=-1)
    d_evoked = np.mean(mean_tar_sch, axis=-1)

    # a_evoked = mean_src_nor[:, fre]
    # b_evoked = mean_src_sch[:, fre]
    # c_evoked = mean_tar_nor[:, fre]
    # d_evoked = mean_tar_sch[:, fre]

    a_evoked_array = evoked_plot(a_evoked, "Normal_src")
    b_evoked_array = evoked_plot(b_evoked, "Schizophrenia_src")
    c_evoked_array = evoked_plot(c_evoked, "Normal_tar")
    d_evoked_array = evoked_plot(d_evoked, "Schizophrenia_tar")

    # 为evoked数据设置电极位置信息
    montage = mne.channels.read_montage("standard_1020")

    a_evoked_array.set_montage(montage)
    b_evoked_array.set_montage(montage)
    c_evoked_array.set_montage(montage)
    d_evoked_array.set_montage(montage)

    fig, ax = plt.subplots(1, 4, figsize=(8, 3))
    mne.viz.plot_topomap(a_evoked_array.data[:, 0], a_evoked_array.info, names=ch_names, show_names=False,
                 outlines='head', show=False, cmap='jet', axes=ax[0])
    mne.viz.plot_topomap(b_evoked_array.data[:, 0], b_evoked_array.info, names=ch_names, show_names=False,
                 outlines='head', show=False, cmap='jet', axes=ax[1])
    mne.viz.plot_topomap(c_evoked_array.data[:, 0], c_evoked_array.info, names=ch_names, show_names=False,
                 outlines='head', show=False, cmap='jet', axes=ax[2])
    mne.viz.plot_topomap(d_evoked_array.data[:, 0], d_evoked_array.info, names=ch_names, show_names=False,
                 outlines='head', show=False, cmap='jet', axes=ax[3])

    for ax, title in zip(ax[:4], ['Normal_src', 'Schizophrenia_src', 'Normal_tar', 'Schizophrenia_tar']):
        ax.set_title(title, fontsize=8)
    plt.suptitle("{} feature: frequency_band{}".format(mode, fre+1))
    plt.show()

    # im, cn = mne.viz.plot_topomap(d_evoked_array.data[:, 0], d_evoked_array.info, names=ch_names, show_names=False,
    #              outlines='head', show=False, cmap='jet', vmin=None, vmax=None)
    # plt.colorbar(im)
    # plt.show()