import sys, traceback, os, re

import torch
import numpy as np
import pandas as pd

from tqdm import trange

from sklearn.metrics import mean_squared_error, mean_absolute_error

from MyPackage import FileLogger
from MyPackage import DataReader
from MyPackage.utils import mean_predictions

from tensorboardX import SummaryWriter
from glob import glob


class Trainer(object):
    def __init__(self,
                 data_path,
                 logger_path,
                 model_name,
                 train_log_interval,
                 valid_log_interval,
                 load_model_name=None,
                 use_script=True,
                 **kwargs):

        """
        Base class for all models. This class implemnts training, validation and prediction
        loops.

        Parameters
        ----------
        data_path : str
            Path to data file

        logger_path : str
            Path to models storage directory

        model_name : str
            Name for current model

        train_log_interval : int
            Train log sampling rate

        valid_log_interval : int
            Validation log samplig rate

        load_model_name : str, optional, default : None
            Path to model name to load

        use_script : boolean, optional, default : True
            If True filelogger initialze as a script.
            If False initilize for notebook
        """

        # Data Reader
        self.datareader = DataReader(data_path,
                                     **kwargs)
        # File Logger
        self.filelogger = FileLogger(logger_path,
                                     model_name,
                                     load_model_name,
                                     use_script)

        # Check cuda availability
        self.use_cuda = torch.cuda.is_available()

        # Variables
        self.logger_path = logger_path
        self.model_name = model_name
        self.train_log_interval = train_log_interval
        self.valid_log_interval = valid_log_interval

        self.model = None
        self.tensorboard = None

    def save(self,
             model_name):
        """
        Save model
        """
        path = self.filelogger.path + '/model_checkpoint/'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model, path + model_name)

    def load(self,
             path_name):
        """
        Load model
        """
        print('Loading file from {}'.format(path_name))
        self.model = torch.load(path_name)
        if self.use_cuda:
            self.model.cuda()

    def train(self,
              patience):

        """
        Training loop to train models

        """

        self.prepare_datareader()
        self.model.apply(self.init_weights)
        self.filelogger.start()
        self.tensorboard = SummaryWriter(self.filelogger.path + '/tensorboard/')

        try:
            training_step = 0
            validation_step = 0

            best_validation_loss = 1000
            validation_loss = 1000
            train_loss = 1000
            best_validation_epoch = 0

            patience_step = 0

            epoch_range = trange(int(self.num_epoch),
                                 desc='1st loop',
                                 unit=' Epochs')

            for epoch in epoch_range:
                batch_train_range = trange(int(self.datareader.train_steps),
                                           desc='2st loop',
                                           unit=' Batch',
                                           leave=True)

                batch_valid_range = trange(int(self.datareader.validation_steps),
                                           desc='2st loop',
                                           unit=' Batch',
                                           leave=True)

                total_train_loss = 0

                for batch_train in batch_train_range:
                    batch_train_range.set_description("Training on %i points --- " % self.datareader.train_length)

                    self.model.train()

                    loss, total_loss = self.training_step()

                    total_train_loss += total_loss

                    batch_train_range.set_postfix(MSE = loss,
                                                  Last_batch_MSE = train_loss,
                                                  Epoch = epoch)

                    self.tensorboard.add_scalar('Training Mean Squared Error loss per batch',
                                                loss,
                                                training_step)

                    self.filelogger.write_train(self.train_log_interval,
                                                training_step,
                                                epoch,
                                                batch_train,
                                                loss)

                    training_step += 1

                train_loss = total_train_loss / (self.datareader.train_length)

                self.tensorboard.add_scalar('Training Mean Squared Error loss per epoch',
                                            train_loss,
                                            epoch)

                total_valid_loss = 0

                for batch_valid in batch_valid_range:
                    batch_valid_range.set_description("Validate on %i points --- " % self.datareader.validation_length)

                    batch_valid_range.set_postfix(Last_Batch_MSE=' {0:.9f} MSE'.format(validation_loss),
                                                  Best_MSE=best_validation_loss,
                                                  Best_Epoch = best_validation_epoch,
                                                  Current_Epoch=epoch)

                    self.model.eval()

                    valid_loss, total_loss = self.evaluation_step()

                    total_valid_loss += total_loss

                    self.tensorboard.add_scalar('Validation Mean Squared Error loss per batch',
                                                valid_loss,
                                                validation_step)

                    self.filelogger.write_valid(self.valid_log_interval,
                                                validation_step,
                                                epoch,
                                                batch_valid,
                                                valid_loss)
                    validation_step += 1

                validation_loss = total_valid_loss / (self.datareader.validation_length)

                self.tensorboard.add_scalar('Validation Mean Squared Error loss per epoch',
                                            validation_loss,
                                            epoch)

                if self.use_scheduler:
                    self.scheduler.step(validation_loss)

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_validation_epoch = epoch + 1
                    patience_step = 0
                    self.save('Model_Checkpoint' + str(epoch + 1) + '_valid_loss_' + str(best_validation_loss) + '.pth')
                else:
                    patience_step += 1
                    if patience_step > patience:
                        print('Train is donne, 3 epochs in a row without improving validation loss!')
                        return best_validation_loss

            print('Train is donne after 10 epochs!')
            return best_validation_loss

        except KeyboardInterrupt:
            if epoch > 0:
                print("Shutdown requested...saving and exiting")
                self.save('Model_save_before_exiting_epoch_' + str(epoch + 1) + '_batch_' + str(
                    batch_train) + '_batch_valid_' + str(batch_valid) + '.pth')
            else:
                print('Shutdown requested!')

        except Exception:
            if epoch > 0:
                self.save('Model_save_before_exiting_epoch_' + str(epoch + 1) + '_batch_' + str(
                    batch_train) + '_batch_valid_' + str(batch_valid) + '.pth')
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)

    def train_cv(self, number_splits, days, patience):
        """
        Train with Cross-Validation

        """

        try:
            mean_score = []
            cv_train_indexes, cv_val_indexes = self.datareader.cross_validation_time_series(number_splits,
                                                                                            days,
                                                                                            self.test_date)

            for model_number in range(number_splits):

                self.model.apply(self.init_weights)

                self.filelogger.start('Fold_Number{0}'.format(model_number + 1))
                self.tensorboard = SummaryWriter(self.filelogger.path + '/tensorboard/')

                self.prepare_datareader_cv(cv_train_indexes[model_number],
                                           cv_val_indexes[model_number])

                training_step = 0
                validation_step = 0

                best_validation_loss = 1000
                validation_loss = 1000
                train_loss = 1000
                best_validation_epoch = 0

                patience_step = 0

                epoch_range = trange(int(self.num_epoch),
                                     desc='1st loop',
                                     unit=' Epochs',
                                     leave=True)

                for epoch in epoch_range:
                    batch_train_range = trange(int(self.datareader.train_steps),
                                               desc='2st loop',
                                               unit=' Batch',
                                               leave=False)

                    batch_valid_range = trange(int(self.datareader.validation_steps),
                                               desc='2st loop',
                                               unit=' Batch',
                                               leave=False)

                    total_train_loss = 0
                    for batch_train in batch_train_range:
                        batch_train_range.set_description("Training on %i points --- " % self.datareader.train_length)

                        self.model.train()

                        loss, total_loss = self.training_step()

                        total_train_loss += total_loss

                        batch_train_range.set_postfix(MSE = loss,
                                                      Last_batch_MSE = train_loss,
                                                      Epoch = epoch)

                        self.tensorboard.add_scalar('Training Mean Squared Error loss per batch',
                                                    loss,
                                                    training_step)

                        self.filelogger.write_train(self.train_log_interval,
                                                    training_step,
                                                    epoch,
                                                    batch_train,
                                                    loss)

                        training_step += 1

                    train_loss = total_train_loss / (self.datareader.train_length)

                    self.tensorboard.add_scalar('Training Mean Squared Error loss per epoch',
                                                train_loss,
                                                epoch)

                    total_valid_loss = 0

                    for batch_valid in batch_valid_range:
                        batch_valid_range.set_description("Validate on %i points --- " % self.datareader.validation_length)

                        batch_valid_range.set_postfix(Last_Batch_MSE=' {0:.9f} MSE'.format(validation_loss),
                                                      Best_MSE=best_validation_loss,
                                                      Best_Epoch=best_validation_epoch,
                                                      Current_Epoch=epoch)

                        self.model.eval()

                        valid_loss, total_loss = self.evaluation_step()

                        total_valid_loss += total_loss

                        self.tensorboard.add_scalar('Validation Mean Squared Error loss per batch',
                                                    valid_loss,
                                                    validation_step)

                        self.filelogger.write_valid(self.valid_log_interval,
                                                    validation_step,
                                                    epoch,
                                                    batch_valid,
                                                    valid_loss)

                        validation_step += 1

                    validation_loss = total_valid_loss / (self.datareader.validation_length)

                    self.tensorboard.add_scalar('Validation Mean Squared Error loss per epoch',
                                                validation_loss,
                                                epoch)

                    if self.use_scheduler:
                        self.scheduler.step(validation_loss)

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_validation_epoch = epoch + 1
                        patience_step = 0
                        self.save('Model_Checkpoint' + str(epoch + 1) + '_valid_loss_' + str(best_validation_loss) + '.pth')
                    else:
                        patience_step += 1
                        if patience_step > patience:
                            break

                mean_score.append(best_validation_loss)

            return np.mean(mean_score)

        except KeyboardInterrupt:
            print('Shutdown requested!')

        except Exception:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)

    def predict(self):

        """
        Prediction loop

        """

        self.prepare_datareader()

        predictions = []
        labels = []

        for batch_test in range(self.datareader.test_steps):
            self.model.eval()

            prediction, Y = self.prediction_step()
            predictions.append(prediction.cpu().data.numpy())
            labels.append(Y)

        return np.concatenate(predictions), np.concatenate(labels)

    def postprocess(self, predictions, labels):

        predictions = self.datareader.normalizer.inverse_transform(predictions)
        labels = self.datareader.normalizer.inverse_transform(labels)

        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)

        target = self.datareader.data.iloc[self.datareader.test_indexes[:-1]]
        results = target.assign(predictions=pd.Series(mean_predictions(predictions), index=target.index).values)

        return results, mse, mae

    def get_best(self, path=None):

        if path is None:
            files = glob(self.filelogger.path + '/model_checkpoint/*')
        else:
            print(path + '/model_checkpoint/*')
            files = glob(path + '/model_checkpoint/*')

        best = 100
        for file in files:
            number = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', file)
            result = float(number[-1])
            if result < best:
                best = result
                best_file = file

        self.load(best_file)

    def init_weights(self):
        raise NotImplementedError

    def prepare_datareader(self):
        raise NotImplementedError

    def prepare_datareader_cv(self):
        raise NotImplementedError

    def traning_step(self):
        raise NotImplementedError

    def evaluation_steo(self):
        raise NotImplementedError

    def prediction_steo(self):
        raise NotImplementedError


if __name__ == "__main__":

    from MyPackage.models import WaveNetContinuosTrainer, RNNTrainer, EncoderDecoderTrainer


    model = WaveNetContinuosTrainer(data_path='/datadrive/wind_power/data/wind_15min.csv',
                                    logger_path='/home/rneves/temp/temp_logger/',
                                    model_name='Test_EncoderDecoder2',
                                    train_log_interval=100,
                                    valid_log_interval=100,
                                    n_residue=2,
                                    n_skip=2,
                                    dilation_depth=2,
                                    n_repeat=1,
                                    number_steps_predict=24,
                                    lr=0.001,
                                    batch_size=256,
                                    num_epoch=1,
                                    target_column='Power',
                                    number_features_input=1,
                                    number_features_output=1,
                                    loss_function='MSE',
                                    optimizer='Adam',
                                    normalizer='Standardization',
                                    use_scheduler=False,
                                    use_script=True,
                                    validation_date='2015-01-01 00:00:00',
                                    test_date='2016-01-01 00:00:00',
                                    index_col=['Date'],
                                    parse_dates=True)
    
    '''

    model = RNNTrainer(data_path='/datadrive/wind_power/data/wind_15min.csv',
                       logger_path='/home/rneves/temp/temp_logger/',
                       model_name='Run_Best_Model',
                       use_script=True,
                       lr=0.001,
                       number_steps_train=2,
                       number_steps_predict=100,
                       batch_size=256,
                       num_epoch=1,
                       hidden_size=4,
                       num_layers=1,
                       cell_type='RNN',
                       kernel_size=3,
                       target_column='Power',
                       validation_date='2015-01-01 00:00:00',
                       test_date='2016-01-01 00:00:00',
                       train_log_interval=100,
                       valid_log_interval=100,
                       use_scheduler=False,
                       normalizer='Standardization',
                       optimizer='Adam',
                       index_col=['Date'],
                       parse_dates=True)
    '''
    '''
    model = EncoderDecoderTrainer(data_path='/datadrive/wind_power/wind_15min.csv',
                                  logger_path='/home/rneves/temp/temp_logger',
                                  model_name='Test_EncoderDecoder',
                                  lr=0.001,
                                  number_steps_train=2,
                                  number_steps_predict=2,
                                  batch_size=256,
                                  num_epoch=1,
                                  hidden_size_encoder=2,
                                  hidden_size_decoder=2,
                                  num_layers=1,
                                  cell_type_encoder='LSTM',
                                  cell_type_decoder='LSTM',
                                  number_features_encoder=1,
                                  number_features_decoder=1,
                                  number_features_output=1,
                                  use_attention=True,
                                  target_column='Power',
                                  validation_date='2015-01-01 00:00:00',
                                  test_date='2016-01-01 00:00:00',
                                  train_log_interval=100,
                                  valid_log_interval=100,
                                  use_scheduler=True,
                                  normalizer='Standardization',
                                  use_script=True,
                                  index_col=['Date'],
                                  parse_dates=True)
    '''

    model.train_cv(2, 90, 2)
    model.train(2)
    model.get_best()
    predictions, labels = model.predict()
    final_df, mse, mae = model.postprocess(predictions, labels)
    model.filelogger.write_results(predictions, labels, final_df, mse, mae)
    print('Done!')
