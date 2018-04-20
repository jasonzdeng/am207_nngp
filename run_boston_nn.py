import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=1, hidden_dim=25):
        super(Model, self).__init__()
        self.linear_input = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList()
        self.num_hidden_layers = hidden_layers
        for i in range(hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.linear_output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear_input(x)
        x = torch.nn.functional.tanh(x)
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
            x = torch.nn.functional.tanh(x)
        x = self.linear_output(x)
        y_pred = x
        return y_pred


class NNR:
    def __init__(self, input_dim, output_dim, hidden_dim=25, hidden_layers=1):
        self._model = Model(input_dim, output_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim)
        self._criterion = nn.MSELoss()
        self._fit_params = dict(lr=0.001, epochs=100, batch_size=64)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._fit_params['lr'])
        self._losses = None
        self._accuracy = None

    def __repr__(self):
        num = 0
        for k, p in self._model.named_parameters():
            numlist = list(p.data.numpy().shape)
            if len(numlist) == 2:
                num += numlist[0] * numlist[1]
            else:
                num += numlist[0]
        return repr(self._model) + "\n" + repr(self._fit_params) + "\nNum Params: {}".format(num)

    def fit(self, X, y):
        # our loss function is cross entropy loss
        # criterion = torch.nn.CrossEntropyLoss(size_average=True)
        # we are optimizing with SGD with a learning rate of 0.01
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)

        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self._fit_params['batch_size'])

        losses = []  # where we'll be storing loss calculations
        batches = 0  # where we'll count the total number of batches
        accuracy = []  # keep track of validation accuracy

        # Training loop
        for epoch in range(self._fit_params['epochs']):
            for i, data in enumerate(train_loader, 0):
                # Forward pass: Compute predicted y by passing x to the model
                predictors, labels = data
                predictors, labels = Variable(predictors.float()), Variable(labels.float())

                pred = self._model.forward(predictors)

                # Compute and store, print loss every 100 cycles
                loss = self._criterion(pred, labels)
                if i % 100 == 0:
                    # print(epoch, i, loss.data[0])
                    losses.append([batches, loss.data[0]])
                    batches += 100

                # Zero gradients, perform a backward pass, and update the weights.
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            accuracy.append([epoch, self.test_accuracy(X, y)])

        # save losses and accuracy as model trains
        self._losses = np.array(losses).T
        self._accuracy = np.array(accuracy).T

    def test_accuracy(self, X, y):
        y_hat = self.predict(X)
        mse = mean_squared_error(y, y_hat)
        return mse

    def set_fit_params(self, *, lr=0.001, epochs=100, batch_size=64, l2_weight=0):
        self._fit_params['batch_size'] = batch_size
        self._fit_params['epochs'] = epochs
        self._fit_params['lr'] = lr
        self._fit_params['l2_weight'] = l2_weight
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._fit_params['lr'],
                                          weight_decay=self._fit_params['l2_weight'])

    def predict(self, X):
        X = Variable(torch.from_numpy(X).float())
        pred = self._model.forward(X)
        return pred.data.numpy()

    def score(self, X, y):
        y_hat = self.predict(X)
        r2 = r2_score(y, y_hat)
        return r2


def run_nn_eval(hidden_dim=25, hidden_layers=8):
    boston = load_boston()

    # split into test and training data
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=444, test_size=.25)
    # scale each predictor to be zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = y_train.reshape(-1, 1)
    # don't leak into test data
    X_test = scaler.transform(X_test)
    y_test = y_test.reshape(-1, 1)

    boston_nnr = NNR(13, 1, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

    boston_nnr.fit(X_train, y_train)

    train_mse = boston_nnr.test_accuracy(X_train, y_train)
    train_r2 = boston_nnr.score(X_train, y_train)
    print('Train MSE: {:.3f}\tTrain R2: {:.3f}'.format(train_mse, train_r2))

    test_mse = boston_nnr.test_accuracy(X_test, y_test)
    test_r2 = boston_nnr.score(X_test, y_test)
    print('Test MSE: {:.3f}\tTest R2: {:.3f}'.format(test_mse, test_r2))


def main():
    run_nn_eval()


if __name__ == '__main__':
    main()
