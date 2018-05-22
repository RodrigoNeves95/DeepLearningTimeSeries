import os, datetime, time

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from functools import wraps

SEED = 1337


def timeit(method):
    @wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = float((te - ts) * 1000)
        else:
            print ('%r  %2.4f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


class DataReader(object):

    def __init__(self,
                 raw_data_path,
                 **kwargs):

        """
        Class to load to a pandas DataFrame, preprocess and create batch generators.

        Parameters
        ----------
        raw_data_path : string
                        Data path. Currently accept csv, parquet, hdf5,
                        pickle, txt and xlsx extension. yo

        kwargs : kwargs to pandas DataFrame loader


        """

        assert os.path.isfile(raw_data_path) is True, \
            'This file do not exist. Please select an existing file'
        assert raw_data_path.lower().endswith(('.csv', '.parquet', '.hdf5', '.pickle', '.txt', '.xlsx')) is True, \
            'This class can\'t handle this extension. Please specify a .csv, .parquet, .hdf5, .pickle extension'

        self.raw_data_path = raw_data_path

        self.loader = self.loader_engine(**kwargs)

        self.data = self.loader()
        self.length = len(self.data)

        self.look_back = None
        self.look_further = None

        self.train_indexes = None
        self.validation_indexes = None
        self.test_indexes = None

        self.train_length = None
        self.validation_length = None
        self.test_length = None

        self.train_steps = None
        self.validation_steps = None
        self.test_steps = None

        self.normalizer = None

    def loader_engine(self, **kwargs):
        """

        Choose the correct pandas loader, accordingly with file extension.

        Parameters
        ----------
        kwargs
            pandas DataFrame loader args

        Returns
        -------
        Pandas DataFrame loader

        """
        if self.raw_data_path.lower().endswith('.csv'):
            return lambda: pd.read_csv(self.raw_data_path, **kwargs)
        elif self.raw_data_path.lower().endswith('.parquet'):
            return lambda: pd.read_parquet(self.raw_data_path, **kwargs)
        elif self.raw_data_path.lower().endswith('.hdf5'):
            return lambda: pd.read_hdf(self.raw_data_path, **kwargs)
        elif self.raw_data_path.lower().endswith(('.pkl', 'pickle')):
            return lambda: pd.read_pickle(self.raw_data_path, **kwargs)
        elif self.raw_data_path.lower().endswith('.txt'):
            return lambda: pd.read_csv(self.raw_data_path, **kwargs)
        elif self.raw_data_path.lower().endswith('.xlsx'):
            return lambda: pd.read_excel(self.raw_data_path, **kwargs)

    def split(self,
              split1,
              split2=None,
              return_df=False):

        """
        Split made based on sets sizes
        Split data in different sets for model training

        Parameters
        ----------
        split1 : float
            Float between 0 and 1 to split dataset in train-validation
            This argument specify the size of training set. The remaining
            if the validation set.

        split2 : float, optional, default : None
            Float between 0 and 1 to split dataset in train-validation-test
            This argument specifies the size of validation set. The remaining
            from split1 + split2 is the test set

        return_df : boolean, option, default : False
            If False return array with indexes position for train, validation, (test)
            If True return DataFrames train, validation, (test)

        """

        arr = np.arange(self.length)

        if split2 is None:

            assert split1 < 1, "split should be smaller than 1"

            split = self.length - int(self.length * split1)

            train_index = arr[:split]
            validation_index = arr[split:]

            if return_df:
                return self.data.iloc[train_index], self.data.iloc[validation_index]
            else:
                return train_index, validation_index

        else:

            assert split1 + split2 < 1, "split1 + split2 should be smaller than 1"

            split_1 = int(self.length * split1)
            split_2 = int(self.length * (split1 + split2))

            train_index = arr[:split_1]
            validation_index = arr[split_1:split_2]
            test_index = arr[split_2:]

            if return_df:
                return self.data.iloc[train_index], self.data.iloc[validation_index], self.data.iloc[test_index]
            else:
                return train_index, validation_index, test_index

    def split_date(self,
                   split_date1,
                   split_date2=None,
                   return_df=False):
        """
            Split made based on time
            Split data in different sets for model training

            Parameters
            ----------
            split_date1 : string
                datetime string to specify train-validation split
                split_date1 is the date where validation set begins

            split_date2 : datetime, optional, default : None
                string to specify train-validation-split
                split_date2 is the date where test set begins

            return_df : boolean, option, default : False
                If False return array with indexes position for train, validation, (test)
                If True return DataFrames train, validation, (test)

        """
        assert isinstance(self.data.index, pd.DatetimeIndex), \
            'Index should be an DatetimeIndex type'

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
            test_range = len(self.data.loc[test_split:])

            train_index = np.arange(self.length - validation_range - test_range)
            validation_index = np.arange(self.length - validation_range - test_range, self.length - test_range)
            test_index = np.arange(self.length - test_range, self.length)

            if return_df:
                return self.data.iloc[train_index], self.data.iloc[validation_index], self.data.iloc[test_index]
            else:
                return train_index, validation_index, test_index

    def split_dataset(self,
                      split1,
                      split2=None,
                      return_df=False):
        """
        Function to decide which split use. If split or split_date.


        """

        if type(split1) == float:
            return self.split(split1=split1, split2=split2, return_df=return_df)
        elif type(split1) == str:
            return self.split_date(split_date1=split1, split_date2=split2, return_df=return_df)
        else:
            NotImplementedError

    def preprocessing_data(self,
                           look_back,
                           look_further,
                           batch_size,
                           validation_split,
                           test_split=None,
                           normalizer=None):

        """
        This function should implement all preprocessing to do in the dataset.
        This function prepares attributes for batch generators.

        Parameters
        ----------
        look_back : int
            Sequence length to use in training

        look_further : int
            Sequence length to predict

        batch_size : int
            Batch Size

        validation_split : float or datetime
            Split for validation set

        test_split : float or datetime, optional, default : None
            Split for test set

        normalizer : string, optional, default : None
            Sklearn preprocessing normalizer. Standarization and Min-Max scaler Implemented
        -------

        """

        self.look_back = look_back
        self.look_further = look_further

        if test_split is None:

            self.train_indexes, self.validation_indexes = self.split_dataset(validation_split)

            self.train_length = len(self.train_indexes)
            self.validation_length = len(self.validation_indexes)

            self.train_steps = round((self.train_length - self.look_back - self.look_further + 1) / batch_size + 0.5)
            self.validation_steps = round((self.validation_length - self.look_further + 1) / batch_size + 0.5)
        else:
            self.train_indexes, self.validation_indexes, self.test_indexes = self.split_dataset(validation_split,
                                                                                                test_split)

            self.train_length = len(self.train_indexes) - 1
            self.validation_length = len(self.validation_indexes) - 1
            self.test_length = len(self.test_indexes) - 1

            self.train_steps = round((len(self.train_indexes) - self.look_back - self.look_further) / batch_size + 0.5)
            self.validation_steps = round((len(self.validation_indexes) - self.look_further) / batch_size + 0.5)
            self.test_steps = round((len(self.test_indexes) - self.look_further) / batch_size + 0.5)

        if normalizer is not None:
            if normalizer == 'Standardization':
                self.normalizer = StandardScaler().fit(self.data.iloc[self.train_indexes])
            elif normalizer == 'MixMaxScaler':
                self.normalizer = MinMaxScaler(feature_range=(-1, 1)).fit(self.data.iloc[self.train_indexes])

    def preprocessing_data_cv(self,
                              look_back,
                              look_further,
                              batch_size,
                              cv_train_indexes,
                              cv_val_indexes,
                              normalizer='Standardization'):

        """
        This function should implement all preprocessing
        to do in the dataset to use in cross-validation process.
        This function prepares attributes to use in batch generators.

        Parameters
        ----------
        look_back : int
            Sequence length to use in training

        look_further : int
            Sequence length to predict

        batch_size : int
            Batch Size

        cv_train_indexes : np.array
            train indexes positions

        cv_val_indexes : np.array
            validation indexes positions

        normalizer : string, optional, default : None
            Sklearn preprocessing normalizer. Standarization and Min-Max scaler Implemented
        -------

        """

        self.look_back = look_back
        self.look_further = look_further

        self.train_indexes, self.validation_indexes = cv_train_indexes, cv_val_indexes

        self.train_length = len(self.train_indexes) - 1
        self.validation_length = len(self.validation_indexes) - 1

        self.train_steps = round((self.train_length - self.look_back - self.look_further) / batch_size + 0.5)
        self.validation_steps = round((self.validation_length - self.look_further) / batch_size + 0.5)

        if normalizer is not None:
            if normalizer == 'Standardization':
                self.normalizer = StandardScaler().fit(self.data.iloc[self.train_indexes])
            elif normalizer == 'MixMaxScaler':
                self.normalizer = MinMaxScaler(feature_range=(-1, 1)).fit(self.data.iloc[self.train_indexes])

    def generator_train(self,
                        batch_size,
                        target,
                        shuffle=True,
                        allow_smaller_batch=True,
                        normalize=True):
        """
        Train batch generator.

        Parameters
        ----------
        batch_size : int

        target : string
            Column name from our target column (column to predict).

        shuffle : boolean, default : True
            If True shuffle data

        allow_smaller_batch : boolean, default : True
            If True last batch from each epoch can be smaller

        normalize : boolean, default : True
            If True apply normalization fo the data

        """

        batch_i = 0
        batch_x = None
        batch_y = None

        assert len(self.train_indexes) > self.look_back - self.look_further, \
            'Train length is too small, since its smaller then self.look_back'
        assert len(self.train_indexes) > batch_size, \
            'Reduce batch_size. Train length is bigger then batch size.'

        indexes = self.train_indexes[:(-self.look_back - self.look_further)]

        while True:
            if shuffle:
                np.random.seed(SEED)
                np.random.shuffle(indexes)

            for position in indexes:
                if batch_x is None:
                    batch_x = np.zeros((batch_size, self.look_back, len(self.data.columns)), dtype='float32')
                    batch_y = np.zeros((batch_size, self.look_further), dtype='float32')

                train_data = self.data.iloc[position:position + self.look_back]
                train_label = self.data[target].iloc[position +
                                                     self.look_back:position + self.look_back + self.look_further]

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

    def generator_validation(self,
                             batch_size,
                             target,
                             normalize=True):
        """
        Validation batch generator.

        Parameters
        ----------
        batch_size : int

        target : string
            Column name from our target column (column to predict).

        normalize : boolean, default : True
            If True apply normalization fo the data

        """

        batch_i = 0
        batch_x = None
        batch_y = None

        assert len(self.validation_indexes) > self.look_back - self.look_further + 1, \
            'Validation length is too small, since its smaller then self.look_back'
        assert len(self.validation_indexes) > batch_size, \
            'Reduce batch_size. Validation length is smaller then batch size.'

        indexes = self.validation_indexes[:(-self.look_further)] - self.look_back

        while True:
            for position in indexes:
                if batch_x is None:
                    batch_x = np.zeros((batch_size, self.look_back, len(self.data.columns)), dtype='float32')
                    batch_y = np.zeros((batch_size, self.look_further), dtype='float32')

                validation_data = self.data.iloc[position:position + self.look_back]
                validation_labels = self.data[target].iloc[position + self.look_back:
                                                           position + self.look_back + self.look_further]

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

            yield batch_x[:batch_i], batch_y[:batch_i]
            batch_x = None
            batch_y = None
            batch_i = 0

    def generator_test(self,
                       batch_size,
                       target,
                       normalize=True):
        """
        Test batch generator.

        Parameters
        ----------
        batch_size : int

        target : string
            Column name from our target column (column to predict).

        normalize : boolean, default : True
            If True apply normalization fo the data

        """

        batch_i = 0
        batch_x = None
        batch_y = None

        assert len(self.test_indexes) > self.look_back - self.look_further + 1, \
            'Validation length is too small, since its smaller then self.look_back'
        assert len(self.test_indexes) > batch_size, \
            'Reduce batch_size. Validation length is smaller then batch size.'

        indexes = self.test_indexes[:(-self.look_further)] - self.look_back

        while True:
            for position in indexes:
                if batch_x is None:
                    batch_x = np.zeros((batch_size, self.look_back, len(self.data.columns)), dtype='float32')
                    batch_y = np.zeros((batch_size, self.look_further), dtype='float32')

                test_data = self.data.iloc[position:position + self.look_back]
                test_labels = self.data[target].iloc[position +
                                                     self.look_back: position + self.look_back + self.look_further]

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

            yield batch_x[:batch_i], batch_y[:batch_i]
            batch_x = None
            batch_y = None
            batch_i = 0

    def cross_validation_time_series(self,
                                     n_splits,
                                     length_split,
                                     test_date=None):
        """
        Sliding window cross-validation. This function prepares the
        indexes for using in the cross-validation.

        Parameters
        ----------
        n_splits : int
            Number of folds to use

        length_split : int
            Size in days for each split

        test_date : datetime, optional
            If dataset has test data use this variable to say where it starts
        Returns
        -------

        """

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
