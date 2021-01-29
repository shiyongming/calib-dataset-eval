import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from argparse import ArgumentParser
import numpy as np
from tools.calcualte_pca_features import feature_pca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random
random.seed(2021)

classes_names = ["abn1",
                 "abn2",
                 "abn3",
                 "abn4",
                 "abn5",
                 "abn6",
                 "abn7",
                 "abn8",
                 "abn9",
                 "abn10"]
r = lambda: random.randint(0, 255)
colors = []
for i in range(len(classes_names) + 1):
    color = ('#%02X%02X%02X' % (r(), r(), r()))
    colors.append(color)

def visualization_features(pca_feature=None,
                           pca_feature_label_file=None,
                           projected_feature=None,
                           projected_feature_label_file=None,
                           dim=2):
    # load labels of pca features
    if pca_feature_label_file is not None:
        pca_feature_labels = np.load(pca_feature_label_file)
        pca_feature_labels = pca_feature_labels.reshape(len(pca_feature_labels))
        if len(pca_feature_labels) != len(pca_feature):
            raise AssertionError("label number of trainset samples does not match the trainset feature number")
    # load labels projected features
    if projected_feature_label_file is not None:
        projected_feature_labels = np.load(projected_feature_label_file)
        projected_feature_labels = projected_feature_labels.reshape(len(projected_feature_labels))
        if len(projected_feature_labels) != len(projected_feature):
            raise AssertionError("label number of testset samples does not match the testset feature number")

    area = np.pi * 2 ** 2
    if(dim == 3):
        fig = plt.figure()
        ax = Axes3D(fig)
        # plot pca features for plot_dim=3
        if (pca_feature_label_file is not None):
            if (len(pca_feature_labels) == len(pca_feature)):
                for cls_name in classes_names:
                    ax.scatter(pca_feature[pca_feature_labels==cls_name,0],
                               pca_feature[pca_feature_labels==cls_name, 1],
                               c=colors[classes_names.index(cls_name)],
                               s=area,
                               alpha=0.3)
        else:
            ax.scatter(pca_feature[:,0],
                       pca_feature[:,1],
                       c="red",
                       s=area,
                       alpha=0.3)

        # plot projected_features for plot_dim=3
        if (projected_feature_label_file is not None):
            if (len(projected_feature_labels) == len(projected_feature)):
                for cls_name in classes_names:
                    ax.scatter(projected_feature[projected_feature_labels==cls_name,0],
                               projected_feature[projected_feature_labels==cls_name, 1],
                               c=colors[classes_names.index(cls_name)],
                               marker='x',
                               alpha=0.3)
        else:
            ax.scatter(projected_feature[:,0],
                       projected_feature[:,1],
                       c="red",
                       marker='x',
                       alpha=0.3)
        plt.savefig(r'features distribution.png', dpi=600)
        plt.show()


    elif(dim==2):
        # plot pca features for plot_dim=2
        if (pca_feature_label_file is not None):
            if (len(pca_feature_labels) == len(pca_feature)):
                for cls_name in classes_names:
                    plt.scatter(pca_feature[pca_feature_labels==cls_name,0],
                                pca_feature[pca_feature_labels==cls_name, 1],
                                c=colors[classes_names.index(cls_name)],
                                s=area,
                                alpha=0.3)
        else:
            plt.scatter(pca_feature[:,0],
                        pca_feature[:,1],
                        c="red",
                        s=area,
                        alpha=0.3)
        # plot projected_features for plot_dim=2
        if (projected_feature_label_file is not None):
            if (len(projected_feature_labels) == len(projected_feature)):
                for cls_name in classes_names:
                    plt.scatter(projected_feature[projected_feature_labels==cls_name,0],
                                projected_feature[projected_feature_labels==cls_name, 1],
                                c=colors[classes_names.index(cls_name)],
                                marker='x',
                                alpha=0.3)
        else:
            plt.scatter(projected_feature[:, 0],
                        projected_feature[:, 1],
                        c="red",
                        marker='x',
                        alpha=0.5)
        plt.savefig(r'features distribution.png', dpi=600)
        plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trainset_np_file', '-tf', default=None, help='numpy file of training set feature')
    parser.add_argument('--trainset_np_array', '-ta', default=None, help='numpy array of training set feature')
    parser.add_argument('--trainset_label_np_file', '-tl', default=None, help='numpy file of training set label')
    parser.add_argument('--calibset_np_file', '-cf', default=None, help='numpy file of testing set feature')
    parser.add_argument('--calibset_np_array', '-ca', default=None, help='numpy array of testing set feature')
    parser.add_argument('--calibset_label_np_file', '-cl', default=None, help='numpy file of testing set label')
    parser.add_argument('--plotdim', '-p', type=int, help='plot dimension')
    args = parser.parse_args()

    # trainset_feature_pca, testset_feature_projected, _ = feature_pca(
    #     npfile_for_cal_pca=r'C:\Users\yoshi\Documents\Codes\MyGithub\calib-dataset-eval\temp_test_data\saved_trainset_features_2.npy',
    #     npfile_for_cal_projection=r'C:\Users\yoshi\Documents\Codes\MyGithub\calib-dataset-eval\temp_test_data\saved_testset_features_2.npy',
    #     dim=2)
    # visualization_features(pca_feature=trainset_feature_pca,
    #                        pca_feature_label_file=r'C:\Users\yoshi\Documents\Codes\MyGithub\calib-dataset-eval\temp_test_data\saved_trainset_labels_4.npy',
    #                        projected_feature = testset_feature_projected,
    #                        dim=2)
    trainset_feature_pca, calib_feature_projected, _ = feature_pca(
        npfile_for_cal_pca=args.trainset_np_file,
        nparray_for_cal_pca=args.trainset_np_array,
        npfile_for_cal_projection=args.calibset_np_file,
        nparray_for_cal_projection=args.calibset_np_array,
        dim=2)
    visualization_features(pca_feature=trainset_feature_pca,
                           pca_feature_label_file=args.trainset_label_np_file,
                           projected_feature = calib_feature_projected,
                           projected_feature_label_file=args.calibset_label_np_file,
                           dim=2)