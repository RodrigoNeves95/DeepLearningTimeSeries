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

        self.prepare_datareader()
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

        try:
            mean_score = []
            cv_train_indexes, cv_val_indexes = self.datareader.cross_validation_time_series(number_splits,
                                                                                            days,
                                                                                            self.test_date)

            for model_number in range(number_splits):

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

        mse = mean_squared_error(predictions, labels)
        mae = mean_absolute_error(predictions, labels)

        target = self.datareader.data.iloc[self.datareader.test_indexes[:-1]]

        target['predictions'] = mean_predictions(predictions)

        return target, mse, mae

    def get_best(self):

        files = glob(self.filelogger.path + '/model_checkpoint/*')

        best = 100
        for file in files:
            number = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', file)
            result = float(number[1])
            if result < best:
                best = result
                best_file = file

        self.load(best_file)
