import os, datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataReader(object):
    def __init__(self,
                 raw_data_path,
                 **kwargs):

        assert os.path.isfile(raw_data_path) is True, 'This file do not exist. Please select an existing file'
        assert raw_data_path.lower().endswith(('.csv', '.parquet', '.hdf5',
                                               '.pickle')) is True, 'This class can\'t handle this extension. Please specify a .csv, .parquet, .hdf5, .pickle extension'

        self.raw_data_path = raw_data_path
        self.loader_engine(**kwargs)
        self.data = self.loader()
        self.length = self.get_length()

    def loader_engine(self, **kwargs):
        if self.raw_data_path.lower().endswith(('.csv')):
            self.loader = lambda: pd.read_csv(self.raw_data_path, **kwargs)
        elif self.raw_data_path.lower().endswith(('.parquet')):
            self.loader = lambda: pd.read_parquet(self.raw_data_path, **kwargs)
        elif self.raw_data_path.lower().endswith(('.hdf5')):
            self.loader = lambda: pd.read_hdf(self.raw_data_path, **kwargs)
        elif self.raw_data_path.lower().endswith(('.pkl', 'pickle')):
            self.loader = lambda: pd.read_pickle(self.raw_data_path, **kwargs)

    def get_length(self):
        return len(self.data)

    def split(self, split1, split2=None, shuffle=False, return_df=False):

        arr = np.arange(self.length)

        if split2 is None:

            split = self.length - int(self.length * split1)

            if return_df:
                return self.data.iloc[arr[:split]], self.data.iloc[arr[split:]]
            else:
                return arr[:split], arr[split:]

        else:
            split1 = self.length - int(self.length * split1) - int(self.length * split2)
            split2 = self.length - int(self.length * split2)

            if return_df:
                return self.data.iloc[arr[:split1]], self.data.iloc[arr[split1:split2]], self.data.iloc[arr[:split2]]
            else:
                return arr[:split1], arr[split1:split2], arr[:split2]

    def split_date(self, split_date1=None, split_date2=None, return_df=False):

        assert isinstance(self.data.index, pd.DatetimeIndex), 'Index should be an DatetimeIndex type'

        if split_date2 is None:
            validation_split = pd.to_datetime(split_date1)
            validation_range = len(self.data.loc[validation_split:])

            train_index = np.arange(self.length - validation_range)
            validation_index = np.arange(self.length - validation_range, self.length)

            if return_df:
                return self.data.iloc[train_index], self.data.iloc[validation_index]
            else:
                return train_index, validation_index

        else:
            validation_split = pd.to_datetime(split_date1)
            test_split = pd.to_datetime(split_date2)

            validation_range = len(self.data.loc[validation_split:test_split]) - 1
            test_range = len(self.data.loc[test_split:]) - 1

            train_index = np.arange(self.length - validation_range - test_range)
            validation_index = np.arange(self.length - validation_range - test_range - 1, self.length - test_range)
            test_index = np.arange(self.length - test_range - 1, self.length + 1)

            if return_df:
                return self.data.iloc[train_index], self.data.iloc[validation_index], self.data.iloc[test_index]
            else:
                return train_index, validation_index, test_index

    def preprocessing_data(self,
                           K,
                           N,
                           batch_size,
                           validation_split,
                           test_split=None,
                           normalizer='Standardization'):

        self.K = K
        self.N = N

        if test_split is None:

            self.train_indexes, self.validation_indexes = self.split_date(validation_split)

            self.train_length = len(self.train_indexes)
            self.validation_length = len(self.validation_indexes)

            self.train_steps = round((self.train_length - self.K - self.N + 1) / batch_size + 0.5)
            self.validation_steps = round((self.validation_length - self.N + 1) / batch_size + 0.5)
        else:
            self.train_indexes, self.validation_indexes, self.test_indexes = self.split_date(validation_split,
                                                                                             test_split)

            self.train_length = len(self.train_indexes) - 1
            self.validation_length = len(self.validation_indexes) - 1
            self.test_length = len(self.test_indexes) - 1

            self.train_steps = round((len(self.train_indexes) - self.K - self.N) / batch_size + 0.5)
            self.validation_steps = round((len(self.validation_indexes) - self.N) / batch_size + 0.5)
            self.test_steps = round(len((self.test_indexes) - self.N) / batch_size + 0.5)

        if normalizer == 'Standardization':
            self.normalizer = StandardScaler().fit(self.data.iloc[self.train_indexes])

        if normalizer == 'MixMaxScaler':
            self.normalizer = MinMaxScaler(feature_range=(-1, 1)).fit(self.data.iloc[self.train_indexes])

    def preprocessing_data_cv(self,
                               K,
                               N,
                               batch_size,
                               cv_train_indexes,
                               cv_val_indexes,
                               normalizer='Standardization'):

        self.K = K
        self.N = N


        self.train_indexes, self.validation_indexes = cv_train_indexes, cv_val_indexes

        self.train_length = len(self.train_indexes) - 1
        self.validation_length = len(self.validation_indexes) - 1

        self.train_steps = round((self.train_length - self.K - self.N) / batch_size + 0.5)
        self.validation_steps = round((self.validation_length - self.N) / batch_size + 0.5)

        if normalizer == 'Standardization':
            self.normalizer = StandardScaler().fit(self.data.iloc[self.train_indexes])

        if normalizer == 'MixMaxScaler':
            self.normalizer = MinMaxScaler(feature_range=(-1, 1)).fit(self.data.iloc[self.train_indexes])

    def generator_train(self,
                        batch_size,
                        target,
                        shuffle=True,
                        allow_smaller_batch=True,
                        normalize=True):

        batch_i = 0
        batch_x = None
        batch_y = None

        assert len(self.train_indexes) > self.K - self.N, 'Train length is too small, since its smaller then self.K'
        assert len(self.train_indexes) > batch_size, 'Reduce batch_size. Train length is bigger then batch size.'

        indexes = self.train_indexes[:(-self.K - self.N)]

        while True:
            if shuffle: np.random.shuffle(indexes)

            for position in indexes:
                if batch_x is None:
                    batch_x = np.zeros((batch_size, self.K, len(self.data.columns)), dtype='float32')
                    batch_y = np.zeros((batch_size, self.N), dtype='float32')

                train_data = self.data.iloc[position:position + self.K]
                train_label = self.data[target].iloc[position + self.K:position + self.K + self.N]

                # if normalize:
                #    batch_x[batch_i] = np.clip(self.normalizer.transform(train_data), -1, 1)
                #    batch_y[batch_i] = np.clip(self.normalizer.transform(train_label.values.reshape(-1,1))[:,0],-1, 1)
                if normalize:
                    batch_x[batch_i] = self.normalizer.transform(train_data)
                    batch_y[batch_i] = self.normalizer.transform(train_label.values.reshape(-1, 1))[:, 0]
                else:
                    batch_x[batch_i] = train_data.values
                    batch_y[batch_i] = train_label.values

                batch_i += 1

                if batch_i == batch_size:
                    yield batch_x, batch_y
                    batch_x = None
                    batch_y = None
                    batch_i = 0

            if allow_smaller_batch:
                yield batch_x[:batch_i], batch_y[:batch_i]
                batch_x = None
                batch_y = None
                batch_i = 0

    def generator_validation(self, batch_size, target, normalize=True):

        batch_i = 0
        batch_x = None
        batch_y = None

        assert len(
            self.validation_indexes) > self.K - self.N + 1, 'Validation length is too small, since its smaller then self.K'
        assert len(
            self.validation_indexes) > batch_size, 'Reduce batch_size. Validation length is smaller then batch size.'

        indexes = self.validation_indexes[:(-self.N)] - self.K

        while True:
            for position in indexes:
                if batch_x is None:
                    batch_x = np.zeros((batch_size, self.K, len(self.data.columns)), dtype='float32')
                    batch_y = np.zeros((batch_size, self.N), dtype='float32')

                validation_data = self.data.iloc[position:position + self.K]
                validation_labels = self.data[target].iloc[position + self.K: position + self.K + self.N]

                if normalize:
                    batch_x[batch_i] = self.normalizer.transform(validation_data)
                    batch_y[batch_i] = self.normalizer.transform(validation_labels.values.reshape(-1, 1))[:, 0]
                else:
                    batch_x[batch_i] = validation_data.values
                    batch_y[batch_i] = validation_labels.values

                batch_i += 1

                if batch_i == batch_size:
                    yield batch_x, batch_y
                    batch_x = None
                    batch_y = None
                    batch_i = 0

            # print('entered here')
            yield batch_x[:batch_i], batch_y[:batch_i]
            batch_x = None
            batch_y = None
            batch_i = 0

    def generator_test(self, batch_size, target, normalize=True):

        batch_i = 0
        batch_x = None
        batch_y = None

        assert len(self.test_indexes) > self.K - self.N + 1, 'Validation length is too small, since its smaller then self.K'
        assert len(self.test_indexes) > batch_size, 'Reduce batch_size. Validation length is smaller then batch size.'

        indexes = self.test_indexes[:(-self.N)] - self.K

        while True:
            # print('entered here1')
            for position in indexes:
                if batch_x is None:
                    batch_x = np.zeros((batch_size, self.K, len(self.data.columns)), dtype='float32')
                    batch_y = np.zeros((batch_size, self.N), dtype='float32')

                test_data = self.data.iloc[position:position + self.K]
                test_labels = self.data[target].iloc[position + self.K: position + self.K + self.N]

                # print(test_data.values, test_labels.values)
                # if normalize:
                #  batch_x[batch_i] = np.clip(self.normalizer.transform(test_data), -1, 1)
                #  batch_y[batch_i] = np.clip(self.normalizer.transform(test_labels.values.reshape(-1,1))[:,0], -1, 1)
                if normalize:
                    batch_x[batch_i] = self.normalizer.transform(test_data)
                    batch_y[batch_i] = self.normalizer.transform(test_labels.values.reshape(-1, 1))[:, 0]

                else:
                    batch_x[batch_i] = test_data.values
                    batch_y[batch_i] = test_labels.values

                batch_i += 1

                if batch_i == batch_size:
                    yield batch_x, batch_y
                    batch_x = None
                    batch_y = None
                    batch_i = 0
            # print('entered here2')
            yield batch_x[:batch_i], batch_y[:batch_i]
            batch_x = None
            batch_y = None
            batch_i = 0

    def cross_validation_time_series(self, n_splits, length_split, test_date=None):

        length = int(self.data.index.get_loc(test_date))

        last_index = self.data.loc[test_date].index[-1]
        validation_index = last_index - datetime.timedelta(days=length_split)

        validation_range = len(self.data.loc[validation_index:last_index]) - 1

        cv = []
        cv_val = []

        for i in range(n_splits):
            train_index = np.arange(length - ((n_splits - i) * validation_range) + 1)

            validation_index = np.arange(length - ((n_splits - i) * validation_range),
                                         length - ((n_splits - 1 - i) * validation_range) + 1)

            cv.append(train_index)
            cv_val.append(validation_index)

        return cv, cv_val

    def generator_train_cv(self, batch_size, target, indexes, shuffle=True, allow_smaller_batch=True):
        batch_i = 0
        batch_x = None
        batch_y = None

        train_length = len(indexes)

        assert train_length > self.K - self.N + 1, 'Train length is too small, since its smaller then self.K'
        assert train_length > batch_size, 'Reduce batch_size. Train length is bigger then batch size.'

        while True:
            if shuffle:
                arr = np.random.shuffle(indexes)

            for position in arr:
                if batch_x is None:
                    batch_x = np.zeros((batch_size, self.K, len(self.data.columns)), dtype='float32')
                    batch_y = np.zeros((batch_size, self.N), dtype='float32')

                batch_x[batch_i] = self.data[position:position + self.K].values
                batch_y[batch_i] = self.data[target][position + self.K:position + self.K + self.N].values

                batch_i += 1

                if batch_i == batch_size:
                    yield batch_x, batch_y
                    batch_x = None
                    batch_y = None
                    batch_i = 0

            if allow_smaller_batch:
                yield batch_x[:batch_i], batch_y[:batch_i]
                batch_x = None
                batch_y = None
                batch_i = 0

    def generator_validation_cv(self, batch_size, target, indexes):

        batch_i = 0
        batch_x = None
        batch_y = None

        validation_length = len(indexes)

        assert validation_length > self.K - self.N + 1, 'Validation length is too small, since its smaller then self.K'
        assert validation_length > batch_size, 'Reduce batch_size. Validation length is smaller then batch size.'

        for position in arr:
            if batch_x is None:
                batch_x = np.zeros((batch_size, self.K, len(self.data.columns)), dtype='float32')
                batch_y = np.zeros((batch_size, self.N), dtype='float32')

            batch_x[batch_i] = self.data.iloc[position:position + self.K].values
            batch_y[batch_i] = self.data[target].iloc[position + self.K: position + self.K + self.N].values

            batch_i += 1

            if batch_i == batch_size:
                print(batch_x.shape)
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_i = 0

        yield batch_x[:batch_i], batch_y[:batch_i]