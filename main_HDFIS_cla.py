import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from lib import FirstTSK
from lib import train_mini_batch, test

# hyper-parameters
train_size = 0.7
learning_rate = 0.001
num_fuzzy_set = 3
max_epoch = 100
batch_size = 64

# load dataset
dataset_name = r'SRBCT'
dataset = torch.load(fr'datasets/{dataset_name}.pt')
sam, label = dataset.sample, dataset.target

# one-hot the label
label = torch.LongTensor(preprocessing.OneHotEncoder().fit_transform(label).toarray())

# split train-test samples
tra_sam, test_sam, tra_label, test_label = train_test_split(sam, label, train_size=train_size)

# preprocessing, linearly normalize the training and test samples into the interval [0,1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(tra_sam)
tra_sam = torch.Tensor(min_max_scaler.transform(tra_sam))
test_sam = torch.Tensor(min_max_scaler.transform(test_sam))

# No. samples, features, and classes
num_tra_sam, num_fea = tra_sam.shape
num_class = tra_label.shape[1]

# init the model
myFirstTSK = FirstTSK(num_fea, num_class, num_fuzzy_set, mf='Gaussian_DMF_sig')

# training and test
# myFirstTSK.trained_param(tra_param='THEN')
train_mini_batch(tra_sam, myFirstTSK, tra_label, learning_rate, max_epoch, batch_size=batch_size, optim_type='Adam')
tra_loss, tra_acc = test(tra_sam, myFirstTSK, tra_label)
test_loss, test_acc = test(test_sam, myFirstTSK, test_label)
print(fr'{dataset_name} dataset, training acc: {tra_acc:.4f}, test acc: {test_acc:.4f}')
