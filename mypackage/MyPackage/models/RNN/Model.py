import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torch.optim as optim

import numpy as np

from MyPackage import Trainer
from .QRNN import QRNN
from .TCN import TemporalConvNet
from .DRNN import DRNN

SEED = 1337

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, number_steps_predict, kernel_size=None, num_layers=1, hidden_size=10, cell_type='LSTM'):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.number_steps_predict = number_steps_predict
        self.encoder_cell = None
        self.cell_type = cell_type
        self.output_size = output_size
        self.kernel_size = kernel_size


        assert self.cell_type in ['LSTM', 'RNN', 'GRU', 'DRNN', 'QRNN', 'TCN'], \
            'Not Implemented, choose on of the following options - ' \
            'LSTM, RNN, GRU, DRNN, QRNN, TCN'

        if self.cell_type == 'LSTM':
            self.encoder_cell = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        if self.cell_type == 'GRU':
            self.encoder_cell = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        if self.cell_type == 'RNN':
            self.encoder_cell = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        if self.cell_type == 'QRNN':
            self.encoder_cell = QRNN(self.input_size, self.hidden_size, self.num_layers, self.kernel_size)
        if self.cell_type == 'DRNN':
            self.encoder_cell = DRNN(self.input_size, self.hidden_size, self.num_layers)  # Batch_First always True
        if self.cell_type == 'TCN':
            self.encoder_cell = TemporalConvNet(self.input_size, self.hidden_size, self.num_layers, self.kernel_size)

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden=None):
        outputs, hidden_state = self.encoder_cell(x, hidden)  # returns output variable - all hidden states for seq_len, hindden state - last hidden state
        outputs = self.output_layer(outputs)
        return outputs

    def predict(self, x, hidden=None):

        if self.cell_type == 'DRNN' or 'QRNN' or 'TCN':  # loop to concat output to input in last position and run all the model again
            predictions = []
            output = x
            seq_len = x.shape[1]
            for step in range(self.number_steps_predict):
                output, hidden_state = self.encoder_cell(x[:, -seq_len:, :])
                result = self.output_layer(output[:, -1, :])
                x = torch.cat([x, result.unsqueeze(1)], dim=1)
                predictions.append(result)
            return torch.stack(predictions, dim=1)[:, :, 0]
        else:
            predictions = []
            for step in range(self.number_steps_predict):
                if step == 0:
                    output, hidden_state = self.encoder_cell(x, hidden)  # returns output variable - all hidden states for seq_len, hindden state - last hidden state
                    result = self.output_layer(output[:, -1, :])
                else:
                    output, hidden_state = self.encoder_cell(result.unsqueeze(1), hidden_state)  # returns output variable - all hidden states for seq_len, hindden state - last hidden state
                    result = self.output_layer(output[:, -1, :])
                predictions.append(result)
            return torch.stack(predictions, dim=1)[:, :, 0]

class RNNTrainer(Trainer):
    def __init__(self,
                 lr,
                 number_steps_train,
                 number_steps_predict,
                 hidden_size,
                 num_layers,
                 cell_type,
                 target_column,
                 batch_size,
                 num_epoch,
                 number_features_input = 1,
                 number_features_output = 1,
                 kernel_size = None,
                 loss_function = 'MSE',
                 optimizer = 'Adam',
                 normalizer = 'Standardization',
                 use_scheduler = False,
                 validation_date = None,
                 test_date = None,
                 **kwargs):

        super(RNNTrainer, self).__init__(**kwargs)

        torch.manual_seed(SEED)

        # Hyper-parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.number_features_input = number_features_input
        self.number_features_output = number_features_output
        self.number_steps_train = number_steps_train
        self.number_steps_predict = number_steps_predict
        self.lr = lr
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.use_scheduler = use_scheduler
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.validation_date = validation_date
        self.test_date = test_date
        self.target_column = target_column

        self.file_name = self.filelogger.file_name

        # Save metadata model
        metadata_key = ['number_steps_train',
                        'number_steps_predict',
                        'cell_type',
                        'hidden_size',
                        'kernel_size',
                        'num_layers',
                        'lr',
                        'batch_size',
                        'num_epoch',
                        'target_column',
                        'validation_date',
                        'test_date']

        metadata_value = [self.number_steps_train,
                          self.number_steps_predict,
                          self.cell_type,
                          self.hidden_size,
                          self.kernel_size,
                          self.num_layers,
                          self.lr,
                          self.batch_size,
                          self.num_epoch,
                          self.target_column,
                          self.validation_date,
                          self.test_date]

        metadata_dict = {}
        for i in range(len(metadata_key)):
            metadata_dict[metadata_key[i]] = metadata_value[i]

        # check if it's to load model or not
        if self.filelogger.load_model is not None:
            self.load(self.filelogger.load_model)
            print('Load model from {}'.format(
                self.logger_path + self.file_name + 'model_checkpoint/' + self.filelogger.load_model))
        else:
            self.model = RNNModel(self.number_features_input,
                                  self.number_features_output,
                                  self.number_steps_predict,
                                  self.kernel_size,
                                  self.num_layers,
                                  self.hidden_size,
                                  self.cell_type)

            self.filelogger.write_metadata(metadata_dict)

        # loss function
        if loss_function == 'MSE':
            self.criterion = nn.MSELoss()

        # optimizer
        if optimizer == 'Adam':
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if optimizer == 'SGD':
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        if optimizer == 'RMSProp':
            self.model_optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        if optimizer == 'Adadelta':
            self.model_optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        if optimizer == 'Adagrad':
            self.model_optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)

        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(self.model_optimizer, 'min', patience=2, threshold=1e-5)

        # check CUDA availability
        if self.use_cuda:
            self.model.cuda()

    def prepare_datareader(self):
        # prepare datareader
        self.datareader.preprocessing_data(self.number_steps_train,
                                           self.number_steps_predict,
                                           self.batch_size,
                                           self.validation_date,
                                           self.test_date,
                                           self.normalizer)
        # Initialize train generator
        self.train_generator = self.datareader.generator_train(self.batch_size,
                                                               self.target_column,
                                                               allow_smaller_batch=True)

        # Initialize validation and test generator
        if self.validation_date is not None:
            self.validation_generator = self.datareader.generator_validation(self.batch_size,
                                                                             self.target_column)

        if self.test_date is not None:
            self.test_generator = self.datareader.generator_test(self.batch_size,
                                                                 self.target_column)

    def prepare_datareader_cv(self, cv_train, cv_val):
        # prepare datareader
        self.datareader.preprocessing_data_cv(self.number_steps_train,
                                           self.number_steps_predict,
                                           self.batch_size,
                                           cv_train,
                                           cv_val,
                                           self.normalizer)
        # Initialize train generator
        self.train_generator = self.datareader.generator_train(self.batch_size,
                                                               self.target_column,
                                                               allow_smaller_batch=True)

        if self.validation_date is not None:
            self.validation_generator = self.datareader.generator_validation(self.batch_size,
                                                                             self.target_column)


    def training_step(self):

        self.model_optimizer.zero_grad()
        loss = 0
        X, Y = next(self.train_generator)
        length = X.shape[0]
        Y = np.concatenate((X[:, 1:, 0], np.expand_dims(Y[:, 0], axis=1)), axis=1)
        X = Variable(torch.from_numpy(X)).float().cuda()
        Y = Variable(torch.from_numpy(Y)).float()

        results = self.model(X)

        loss = self.criterion(results, Y.unsqueeze(2).cuda())

        loss.backward()
        self.model_optimizer.step()

        return loss.data[0], loss.data[0] * length

    def evaluation_step(self):

        X, Y = next(self.validation_generator)
        length = X.shape[0]
        Y = np.concatenate((X[:, 1:, 0], np.expand_dims(Y[:, 0], axis=1)), axis=1)
        X = Variable(torch.from_numpy(X), requires_grad=False, volatile=True).float().cuda()
        Y = Variable(torch.from_numpy(Y), requires_grad=False, volatile=True).float()

        results = self.model(X)

        valid_loss = self.criterion(results, Y.unsqueeze(2).cuda())

        return valid_loss.data[0], valid_loss.data[0] * length

    def prediction_step(self):

        X, Y = next(self.test_generator)
        X = Variable(torch.from_numpy(X), requires_grad=False, volatile=True).float().cuda()

        results = self.model.predict(X)

        return results, Y