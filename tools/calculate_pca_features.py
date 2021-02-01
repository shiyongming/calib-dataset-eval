import numpy as np
from sklearn.decomposition import PCA

def feature_pca(npfile_for_cal_pca=None,
                nparray_for_cal_pca=None,
                npfile_for_cal_projection=None,
                nparray_for_cal_projection=None,
                dim=2):
    # load trainset features
    if (npfile_for_cal_pca is not None) and (nparray_for_cal_pca is not None):
        raise AssertionError("Numpy file and numpy array can't be given at the same time.")
    elif nparray_for_cal_pca is not None:
        data_for_cal_pca = nparray_for_cal_pca
    elif npfile_for_cal_pca is not None:
        data_for_cal_pca = np.load(npfile_for_cal_pca)
    else:
        raise AssertionError("Numpy file or numpy array, at least one of them should be given.")
    # reshape the feature into [len(trainset_feature), flatten_feature_dim] for pca calculation
    reshaped_data_for_pca = data_for_cal_pca.reshape(len(data_for_cal_pca), -1)
    # print shape for debug
    # print("trainset feature shape is:", trainset_feature.shape)
    print("trainset reshaped feature shape is:", reshaped_data_for_pca.shape)

    # load testset features
    if (npfile_for_cal_projection is not None) and (nparray_for_cal_projection is not None):
        raise AssertionError("Numpy file and numpy array can't be given at the same time.")
    elif nparray_for_cal_projection is not None:
        data_for_cal_projection = nparray_for_cal_projection
    elif npfile_for_cal_projection is not None:
        data_for_cal_projection = np.load(npfile_for_cal_projection)
    else:
        raise AssertionError("Numpy file or numpy array, at least one of them should be given.")
    # reshape the feature into [len(trainset_feature), flatten_feature_dim] for projection calculation
    reshaped_data_for_projection = data_for_cal_projection.reshape(len(data_for_cal_projection), -1)
    # print shape for debug
    # print("testset feature shape is:", testset_feature.shape)
    print("testset reshaped feature shape is:", reshaped_data_for_projection.shape)

    # calculate pca of trainset features
    pca = PCA(n_components=dim)
    pca = pca.fit(reshaped_data_for_pca)
    feature_1_pca = pca.transform(reshaped_data_for_pca)
    print("trainset feature shape after PCA:", feature_1_pca.shape)
    print("explained_variance of trainset features:", pca.explained_variance_)
    print("explained_variance_ratio of trainset features:", pca.explained_variance_ratio_)

    # calculate the projection of testset features on pca space of trainset features
    Vt = pca.components_
    # print("Vt shape is:", Vt.shape)
    feature_2_projected = np.dot((reshaped_data_for_projection - pca.mean_.T), Vt.T)
    print("shape of the projected testset features on trainset pca space", feature_2_projected.shape)

    return feature_1_pca, feature_2_projected, Vt


if __name__ == '__main__':
    feature_pca(
        npfile_for_cal_pca=r'C:\Users\yoshi\Documents\Codes\MyGithub\calib-dataset-eval\temp_test_data\saved_trainset_features_2.npy',
        # trainset_label_np_file=r'C:\Users\yoshi\Documents\Codes\MyGithub\calib-dataset-eval\temp_test_data\saved_trainset_labels_4.npy',
        npfile_for_cal_projection=r'C:\Users\yoshi\Documents\Codes\MyGithub\calib-dataset-eval\temp_test_data\saved_testset_features_2.npy',
        dim=2)
