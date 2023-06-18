import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from lib import FirstTSK
from lib import train_mini_batch, test

# hyper-parameters
train_size = 0.7
learning_rate = 0.0001
num_fuzzy_set = 3
max_epoch = 100
batch_size = 64

# load dataset
dataset_name = r'DrivFace_reg'
dataset = torch.load(fr'datasets/{dataset_name}.pt')
sam, tar = dataset.sample, dataset.target

# split train-test samples
tra_sam, test_sam, tra_tar, test_tar = train_test_split(sam, tar, train_size=train_size)

# preprocessing, linearly normalize the training and test samples into the interval [0,1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(tra_sam)
tra_sam = torch.Tensor(min_max_scaler.transform(tra_sam))
test_sam = torch.Tensor(min_max_scaler.transform(test_sam))

# in regression, the actual outputs need to be normalized, into the interval [0,1]
tra_tar_raw = tra_tar.clone()
min_max_scaler_target = preprocessing.MinMaxScaler(feature_range=(0, 1))
tra_tar = torch.Tensor(min_max_scaler_target.fit_transform(tra_tar))

# No. samples, features and response variables
num_tra_sam, num_fea = tra_sam.shape
tar_dim = tra_tar.shape[1]

# init the model
myFirstTSK = FirstTSK(num_fea, tar_dim, num_fuzzy_set, mf='Gaussian_DMF_sig')

# training and test
train_mini_batch(tra_sam, myFirstTSK, tra_tar, learning_rate, max_epoch, batch_size=batch_size, optim_type='Adam')
tra_rmse = test(tra_sam, myFirstTSK, tra_tar_raw, task='regression', pre_sca=min_max_scaler_target)
test_rmse = test(test_sam, myFirstTSK, test_tar, task='regression', pre_sca=min_max_scaler_target)
print(fr'{dataset_name} dataset, training RMSE: {tra_rmse:.4f}, test RMSE: {test_rmse:.4f}')
