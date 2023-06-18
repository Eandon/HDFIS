import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics


def train_mini_batch(model_input, model, target_output, learning_rate, max_epoch, batch_size=1,
                     optim_type='SGD', gpu=False):
    """
    train the model
    :param model_input:
    :param model:
    :param target_output:
    :param learning_rate:
    :param batch_size:
    :param max_epoch:
    :param optim_type: optimizer, {'SGD' (default), 'Adam'}
    :param gpu: gpu or not, {Ture, False (default)}
    :return: 训练好的模型
    """
    if gpu:
        device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # go to gpu
        model.to(device_gpu)
        model_input, target_output = model_input.to(device_gpu), target_output.to(device_gpu)

    # Convert to 64-bit floating point
    model_input = model_input.double()
    target_output = target_output.double()

    # Construct the data for mini-batch
    dataloader = DataLoader(TensorDataset(model_input, target_output), batch_size=batch_size, shuffle=True)

    # define loss and optimizer
    criterion = mse_loss_fun
    if optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid value for optim_type: '{}'".format(optim_type))

    # iterate and update
    loss_epoch = []  # Record the loss, one loss for one epoch
    for epoch in range(max_epoch):
        loss_iter = []  # Record the loss, one loss for one iteration
        for model_input_batch, target_output_batch in dataloader:
            # compute model outputs
            model_output_batch, _ = model.forward(model_input_batch)

            # compute loss
            loss = criterion(model_output_batch, target_output_batch)
            loss_iter.append(loss.data)

            # iterate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print
        loss_epoch.append(torch.tensor(loss_iter).mean())
        print('{}-th epoch, training loss: {:.4f}'.format(epoch, loss_epoch[-1]))

    if gpu:
        # back to cpu
        device_cpu = torch.device("cpu")
        model.to(device_cpu)

    return torch.tensor(loss_epoch)


def test(model_input, model, target_output, task='classification', pre_sca=None):
    """
    test the model
    :param model_input:
    :param model:
    :param target_output:
    :param task: tsak type, {classification(default),regression}
    :param pre_sca: preprocessing scaler for target outputs
            only task='regression'
    :return: performance metric, loss or acc
    """
    model_input = model_input.double()
    target_output = target_output.double()

    # model outputs on test samples
    model_output, _ = model.forward(model_input)

    # define loss fun
    criterion = mse_loss_fun

    if task == 'classification':
        # classification, compute loss and acc
        loss = criterion(model_output, target_output)
        acc = metrics.accuracy_score(target_output.argmax(dim=1), model_output.argmax(dim=1))
        return loss, acc
    elif task == 'regression':
        # regression, de-normalized model outputs, only compute loss
        model_output = torch.DoubleTensor(pre_sca.inverse_transform(model_output.data))
        # loss = criterion(model_output, target_output)
        rmse_loss = nn.MSELoss()(model_output, target_output).sqrt()
        return rmse_loss


def mse_loss_fun(y: torch.tensor = None, z: torch.tensor = None):
    """
    self-defined mse loss
    :param y: model outputs
    :param z: target outputs
    :return: mse loss
    """
    return ((y - z) ** 2).sum() / (2 * y.shape[0])
