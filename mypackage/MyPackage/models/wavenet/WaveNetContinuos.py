import time, sys, os, traceback

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, mean_squared_error

from MyPackage import FileLogger
from MyPackage import DataReader
from MyPackage.utils import *

from tensorboardX import SummaryWriter



class WaveNetModelContinuos(nn.Module):
    def __init__(self, number_features, n_residue=32, n_skip=512, dilation_depth=10, n_repeat=5):
        super(WaveNetModelContinuos, self).__init__()

        self.dilation_depth = dilation_depth
        self.number_features = number_features

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
        else:
            receptive_field += filter_width - 1
            self.receptive_field = receptive_field
        return receptive_field

    def generate_slow(self, input, n=100):
        res = input
        for _ in range(n):
            x = res[:, -self.receptive_field:, :]
            y = self.forward(x)
            i = y.permute(0, 2, 1)
            del (y)
            res = torch.cat((res, i[:, 0, :]), dim=1)
        return res

    def predict(self, input):
        res = input
        x = res[:, -self.receptive_field:, :]
        y = self.forward(x)
        predictions = y.permute(0, 2, 1)
        return predictions

    def generate(self, input=None, n=100, temperature=None, estimate_time=False):

        output = self.preprocess(input)
        output_buffer = []

        for s, t, skip_scale, residue_scale, d in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale,
                                                      self.residue_scale, self.dilations):
            output, _ = self.residue_forward(output, s, t, skip_scale, residue_scale)
            sz = 1 if d == 2 ** (self.dilation_depth - 1) else d * 2
            output_buffer.append(output[:, :, -sz - 1:-1])


        res = input.data.tolist()
        for i in range(n):
            output = Variable(torch.LongTensor(res[-2:]))
            output = self.preprocess(output)
            output_buffer_next = []
            skip_connections = []  # save for generation purposes
            for s, t, skip_scale, residue_scale, b in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale,
                                                          self.residue_scale, output_buffer):
                output, residue = self.residue_forward(output, s, t, skip_scale, residue_scale)
                output = torch.cat([b, output], dim=2)
                skip_connections.append(residue)
                if i % 100 == 0:
                    output = output.clone()
                output_buffer_next.append(output[:, :, -b.size(2):])
            output_buffer = output_buffer_next
            output = output[:, :, -1:]
            # sum up skip connections
            output = sum(skip_connections)
            output = self.postprocess(output)
            if temperature is None:
                _, output = output.max(dim=1)
            else:
                output = output.div(temperature).exp().multinomial(1).squeeze()
            res.append(output.data[-1])
        return res


class WaveNetContinuosTrainer(object):
    def __init__(self,
                 data_path,
                 logger_path,
                 name,
                 n_residue,
                 n_skip,
                 dilation_depth,
                 n_repeat,
                 number_steps_predict,
                 lr,
                 batch_size,
                 num_epoch,
                 target_column,
                 number_features,
                 validation_date=None,
                 test_date=None,
                 load_model_name=None,
                 **kwargs):

        metadata_key = ['n_residue', 'n_skip', 'dilation_depth', 'n_repeat', 'number_steps_predict', 'lr', 'batch_size',
                        'num_epoch', 'target_column', 'number of features', 'validation_date', 'test_date']
        metadata_value = [n_residue, n_skip, dilation_depth, n_repeat, number_steps_predict, lr, batch_size, num_epoch,
                          target_column, number_features, validation_date, test_date]

        metadata_dict = {}
        for i in range(len(metadata_key)):
            metadata_dict[metadata_key[i]] = metadata_value[i]

        # Data Reader
        self.datareader = DataReader(data_path, **kwargs)
        # File Logger
        self.filelogger = FileLogger(logger_path, name, load_model_name)
        # Tensorboard
        self.tensorboard = SummaryWriter(logger_path + name + '/tensorboard/')
        # model
        self.model = WaveNetModelContinuos(number_features, n_residue, n_skip, dilation_depth,
                                           n_repeat)  # model parameters here

        # Hyper-parameters
        self.number_steps_predict = number_steps_predict
        self.lr = lr
        self.batch_size = batch_size
        self.num_epoch = num_epoch

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Loss function
        self.criterion = nn.MSELoss()

        # Check cuda availability
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model.cuda()

        # Check if we want to load previous model
        self.logger_path = logger_path
        self.file_name = self.filelogger.file_name

        if self.filelogger.load_model is not None:
            self.load(self.filelogger.load_model)
            print('Load model from {}'.format(
                self.logger_path + self.file_name + 'model_checkpoint/' + self.filelogger.load_model))
        else:
            self.filelogger.write_metadata(metadata_dict)

        # Prepare variables(infer how many steps we should look back for example) and generators for training, validation and test loop
        self.datareader.preprocessing_data(self.model.calculate_receptive_field(self.number_steps_predict),
                                           self.number_steps_predict, self.batch_size,
                                           validation_date,
                                           test_date)

        self.train_generator = self.datareader.generator_train(self.batch_size, target_column, allow_smaller_batch=True)

        if validation_date is not None:
            self.validation_generator = self.datareader.generator_validation(self.batch_size, target_column)
        if test_date is not None:
            self.test_generator = self.datareader.generator_test(self.batch_size, target_column)

    def save(self, model_name):
        path = self.logger_path + self.file_name + '/model_checkpoint/'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model, path + model_name)

    def load(self, path_name):
        self.model = torch.load(path_name)
        if self.use_cuda:
            self.model.cuda()

    def train(self):

        try:

            train_loss_batch = []
            validation_loss_batch = []

            training_step = 0
            validation_step = 0

            train_log_interval = 10
            validation_log_interval = 10

            train_time_elapsed = 0
            validation_time_elapsed = 0

            best_loss = 100000

            for epoch in range(self.num_epoch):
                total_train_loss = 0
                for batch_train in range(2):
                    self.model.train()
                    begin = time.time()

                    self.optimizer.zero_grad()

                    loss = 0

                    X, Y = next(self.train_generator)

                    X = Variable(torch.from_numpy(X)).float().cuda()
                    Y = Variable(torch.from_numpy(Y)).float().cuda()

                    results = self.model(X)

                    loss = self.criterion(results, Y.unsqueeze(1))

                    loss.backward()
                    self.optimizer.step()

                    loss = loss.data[0]
                    total_train_loss += loss
                    train_loss_batch.append(loss)

                    self.tensorboard.add_scalar('Training Mean Squared Error loss per batch', loss, training_step)
                    time_batch = begin - time.time()
                    train_time_elapsed += time_batch
                    self.filelogger.write_train(train_log_interval, (training_step), (epoch), (batch_train), (loss),
                                                (train_time_elapsed), 'Mean Squared Error')

                    training_step += 1

                train_loss = total_train_loss / (self.datareader.train_steps)  # this is not calculated correctly
                self.tensorboard.add_scalar('Training Mean Squared Error loss per epoch', train_loss, epoch)

                total_valid_loss = 0
                for batch_valid in range(self.datareader.validation_steps):
                    self.model.eval()
                    valid_begin = time.time()

                    X, Y = next(self.validation_generator)

                    X = Variable(torch.from_numpy(X), requires_grad=False).float().cuda()
                    Y = Variable(torch.from_numpy(Y), requires_grad=False).float().cuda()

                    results = self.model(X)

                    valid_loss = self.criterion(results, Y.unsqueeze(1))

                    valid_loss = valid_loss.data[0]

                    total_valid_loss += valid_loss

                    validation_loss_batch.append(valid_loss)

                    self.tensorboard.add_scalar('Validation Mean Squared Error loss per batch', valid_loss,
                                                validation_step)

                    valid_time_batch = valid_begin - time.time()
                    validation_time_elapsed += valid_time_batch
                    self.filelogger.write_valid(validation_log_interval, (validation_step), (epoch), (batch_valid),
                                                (valid_loss), (validation_time_elapsed), 'Mean Squared Error')

                    validation_step += 1

                validation_loss = total_valid_loss / (
                self.datareader.validation_steps)  # this is not calculated correctly
                self.tensorboard.add_scalar('Validation CMean Squared Error loss per epoch', validation_loss, epoch)

                if validation_loss < best_loss:
                    best_loss = validation_loss

                    print('Validation loss improved!')
                    self.save('Wavenet_checkpoint_epoch_' + str(epoch + 1) + '_valid_loss_' + str(best_loss) + '.pth')
                else:
                    print('Validation loss did not improved!')

        except KeyboardInterrupt:
            print("Shutdown requested...saving and exiting")
            self.filelogger.write_train(train_log_interval, (training_step), (epoch), (batch_train), (loss),
                                        (train_time_elapsed), 'Cross_entropy')
            self.filelogger.write_valid(validation_log_interval, (validation_step), (epoch), (batch_valid),
                                        (valid_loss), (validation_time_elapsed), 'Cross_entropy')
            self.save('Wavenet_save_before_exiting_epoch_' + str(epoch + 1) + '_batch_' + str(
                batch_train) + '_batch_valid_' + str(batch_valid) + '.pth')
        except Exception:
            self.filelogger.write_train(train_log_interval, (training_step), (epoch), (batch_train), (loss),
                                        (train_time_elapsed), 'Cross_entropy')
            self.filelogger.write_valid(validation_log_interval, (validation_step), (epoch), (batch_valid),
                                        (valid_loss), (validation_time_elapsed), 'Cross_entropy')
            self.save('Wavenet_save_before_exiting_epoch_' + str(epoch + 1) + '_batch_' + str(
                batch_train) + '_batch_valid_' + str(batch_valid) + '.pth')
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)

    def predict_generating(self, n_steps=None):
        if n_steps is None:
            n_steps = self.datareader.N

        predictions = []
        labels = []
        for batch_test in range(self.datareader.test_steps):
            self.model.eval()
            test_begin = time.time()

            total_testloss = 0
            X, Y = next(self.test_generator)

            X = Variable(torch.from_numpy(X), requires_grad=False, volatile=True).float().cuda()
            prediction = self.model.generate_slow(X, n_steps)
            predictions.append(
                prediction.cpu().data.numpy())  # turn on numpy again // process here ? retun label also? too many decisions too little time
            labels.append(Y)

        return np.concatenate(predictions), np.concatenate(labels)

    def predict(self):
        predictions = []
        labels = []
        for batch_test in range(self.datareader.test_steps):
            self.model.eval()
            test_begin = time.time()

            total_testloss = 0
            X, Y = next(self.test_generator)

            X = Variable(torch.from_numpy(X), requires_grad=False, volatile=True).float().cuda()

            prediction = self.model.predict(X)
            predictions.append(
                prediction.cpu().data.numpy())  # turn on numpy again // process here ? retun label also? too many decisions too little time
            labels.append(Y)

        return np.concatenate(predictions), np.concatenate(labels)

    def postprocess(self, predictions, labels):

        predictions = self.datareader.normalizer.inverse_transform(predictions[:, :, 0])
        labels = self.datareader.normalizer.inverse_transform(labels)

        mse = mean_squared_error(predictions, labels)
        mae = mean_absolute_error(predictions, labels)

        target = self.datareader.data.iloc[self.datareader.test_indexes]

        return mean_predictions(predictions), target, mse, mae

