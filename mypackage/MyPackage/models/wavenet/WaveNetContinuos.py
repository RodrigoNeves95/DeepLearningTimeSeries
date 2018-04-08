import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torch.optim as optim

import numpy as np

from MyPackage import Trainer

SEED = 1337

class WaveNetModelContinuos(nn.Module):
    def __init__(self, number_features, number_steps_predict, n_residue=32, n_skip=512, dilation_depth=10, n_repeat=5):
        super(WaveNetModelContinuos, self).__init__()

        self.dilation_depth = dilation_depth
        self.number_features = number_features
        self.number_steps_predict = number_steps_predict

        self.dilations = [2 ** i for i in range(dilation_depth)] * n_repeat

        self.acummulated_length = np.cumsum(self.dilations)

        self.from_input = nn.Conv1d(in_channels=number_features, out_channels=n_residue, kernel_size=1)

        self.conv_sigmoid = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
             for d in self.dilations])

        self.conv_tanh = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
             for d in self.dilations])

        self.skip_scale = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_skip, kernel_size=1)
             for d in self.dilations])

        self.residue_scale = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=1)
             for d in self.dilations])

        self.conv_post_1 = nn.Conv1d(in_channels=n_skip, out_channels=n_skip, kernel_size=1)

        self.conv_post_2 = nn.Conv1d(in_channels=n_skip, out_channels=1, kernel_size=1)

    def forward(self, input):

        output = input.permute(0, 2, 1)
        output = self.from_input(output)
        skip_connections = []
        for s, t, skip_scale, residue_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale,
                                                   self.residue_scale):
            output, skip = self.residue_forward(output, s, t, skip_scale, residue_scale)
            skip_connections.append(skip)
        output = sum([s[:, :, -output.size(2):] for s in skip_connections])
        output = self.postprocess(output)
        return output

    def postprocess(self, input):
        output = nn.functional.elu(input)
        output = self.conv_post_1(output)
        output = nn.functional.elu(output)
        output = self.conv_post_2(output)
        return output

    def residue_forward(self, input, conv_sigmoid, conv_tanh, skip_scale, residue_scale):
        output = input
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        output = nn.functional.sigmoid(output_sigmoid) * nn.functional.tanh(output_tanh)
        skip = skip_scale(output)
        output = residue_scale(output)
        output = output + input[:, :, -output.size(2):]
        return output, skip

    def calculate_receptive_field(self, output_filter_width):
        filter_width = 2
        scalar_input = True

        receptive_field = (filter_width - 1) * sum(self.dilations) + 1
        if scalar_input:
            receptive_field += output_filter_width - 1
            self.receptive_field = receptive_field
        return receptive_field

    def calculate_output_receptive_field(self, input_filter_width):

        self.output_receptive_field = input_filter_width - sum(self.dilations)

        return self.output_receptive_field

    def predict(self, input):
        res = input
        for _ in range(self.number_steps_predict):
            x = res[:, -self.receptive_field:, :]
            y = self.forward(x)
            i = y.permute(0, 2, 1)
            del (y)
            res = torch.cat((res, i[:, -1:, :]), dim=1)
        return res[:, -self.number_steps_predict:, 0]


class WaveNetContinuosTrainer(Trainer):
    def __init__(self,
                 n_residue,
                 n_skip,
                 dilation_depth,
                 n_repeat,
                 number_steps_predict,
                 lr,
                 batch_size,
                 num_epoch,
                 target_column,
                 number_features_input=1,
                 number_features_output=1,
                 loss_function='MSE',
                 optimizer='Adam',
                 normalizer='Standardization',
                 use_scheduler=False,
                 validation_date=None,
                 test_date=None,
                 load_model_name=None,
                 **kwargs):

        super(WaveNetContinuosTrainer, self).__init__(**kwargs)

        torch.manual_seed(SEED)

        self.n_residue = n_residue
        self.n_skip = n_skip
        self.dilation_depth = dilation_depth
        self.n_repeat = n_repeat
        self.number_steps_predict = number_steps_predict
        self.lr = lr
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.target_column = target_column
        self.number_features_input = number_features_input
        self.number_features_output = number_features_output
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.use_scheduler = use_scheduler
        self.validation_date = validation_date
        self.test_date = test_date
        self.load_model_name = load_model_name


        # check if it's to load model or not
        if self.filelogger.load_model is not None:
            self.load(self.filelogger.load_model)
            print('Load model from {}'.format(
                self.logger_path + self.file_name + 'model_checkpoint/' + self.filelogger.load_model))
        else:
            self.model = WaveNetModelContinuos(self.number_features_input,
                                               self.number_steps_predict,
                                               self.n_residue,
                                               self.n_skip,
                                               self.dilation_depth,
                                               self.n_repeat)

            self.number_steps_train = self.model.calculate_receptive_field(self.number_steps_predict)

            metadata_key = ['n_residue',
                            'n_skip',
                            'dilation_depth',
                            'n_repeat',
                            'number_steps_predict',
                            'number_steps_train',
                            'lr',
                            'batch_size',
                            'num_epoch',
                            'target_column',
                            'validation_date',
                            'test_date']

            metadata_value = [self.n_residue,
                              self.n_skip,
                              self.dilation_depth,
                              self.n_repeat,
                              self.number_steps_predict,
                              self.number_steps_train,
                              self.lr,
                              self.batch_size,
                              self.num_epoch,
                              self.target_column,
                              self.validation_date,
                              self.test_date]

            metadata_dict = {}
            for i in range(len(metadata_key)):
                metadata_dict[metadata_key[i]] = metadata_value[i]

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
        Y = Y[:, -self.number_steps_predict:]
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
        X = Variable(torch.from_numpy(X), requires_grad=False, volatile=True).float().cuda()
        Y = Variable(torch.from_numpy(Y), requires_grad=False, volatile=True).float().cuda()

        results = self.model.predict(X)

        valid_loss = self.criterion(results, Y.unsqueeze(2))

        return valid_loss.data[0], valid_loss.data[0] * length

    def prediction_step(self):

        X, Y = next(self.test_generator)
        X = Variable(torch.from_numpy(X), requires_grad=False, volatile=True).float().cuda()

        results = self.model.predict(X)

        return results, Y