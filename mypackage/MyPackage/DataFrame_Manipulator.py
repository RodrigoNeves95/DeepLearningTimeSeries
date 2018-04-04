import copy, os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrame(object):

    """
    Minimal pd.DataFrame analog for handling n-dimensional numpy matrices with additional
    support for shuffling, batching, and train/test splitting.

    -------------------------------------------------------
    Args:
        columns : list
            List of names corresponding to the matrices in data.

        data : numpy file
            List of n-dimensional data matrices ordered in correspondence with columns.
            All matrices must have the same leading dimension.  Data can also be fed a list of
            instances of np.memmap, in which case RAM usage can be limited to the size of a
            single batch.
    """

    def __init__(self, columns, data):

        assert len(columns) == len(data), "columns length does not match data length" #Check if number of columns match the number of data matrices

        lengths = [matrix.shape[0] for matrix in data]
        assert len(set(lengths)) == 1, "all matrices in data must have same first dimension" #The first dimension must be the same for all the data matrices


        self.length = lengths[0]
        self.columns = columns
        self.data =  data
        self.dict = dict(zip(self.columns, self.data)) #Make a dictionary to make the mapping of column to the correspondent data matrices
        self.idx = np.arange(self.length) #Indexes of data matrices

    def shapes(self):
        """Get shapes of data matrices"""
        return pd.Series(dict(zip(self.columns, [matrix.shape for matrix in self.data])))

    def dtypes(self):
        """Get data type of data matrices"""
        return pd.Series(dict(zip(self.columns, [matrix.dtype for matrix in self.data])))

    def shuffle(self):
        """Randomly shuffle the mapping of indexes from the data matrices"""
        np.random.shuffle(self.idx)

    def train_test_split(self, train_size, random_state=np.random.randint(10000)):
        """
        Function to make train-test split for all data matrices

        -------------------------------------------------------
        Args:
            train_size : float, int, or None (default is None)
                If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the train split. If
                int, represents the absolute number of train samples. If None,
                the value is automatically set to the complement of the test size

            random_state : int or RandomState
                Pseudo-random number generator state used for random sampling

        -------------------------------------------------------
        Returns:
            train_df: DataFrame class
                Train DataFrame

            test_df: DataFrame class
                Test DataFrame
        """
        train_idx, test_idx = train_test_split(self.idx, train_size=train_size, random_state=random_state)
        train_df = DataFrame(copy.copy(self.columns), [mat[train_idx] for mat in self.data])
        test_df = DataFrame(copy.copy(self.columns), [mat[test_idx] for mat in self.data])
        return train_df, test_df

    def batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        """
        Batch_generator for DataFrame class

        -------------------------------------------------------
        Args:
            batch_size : int
                Number of samples to use on mini-batch

            shuffle : boolean
                If shuffle is True then data is shuffled if False do nothing

            num_epochs : int
                Number of epochs

            allow_smaller_final_batch : boolean
                If True is possible to have a smaller batch if dimensions don't match

        -------------------------------------------------------
        yield:
            batch : DataFrame class
                Processed batch generated
        """

        epoch_num = 0
        while epoch_num < num_epochs:
            if shuffle:
                self.shuffle()

            for i in range(0, self.length + 1, batch_size):
                batch_idx = self.idx[i: i + batch_size]
                if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                    break
                yield DataFrame(columns=copy.copy(self.columns), data=[mat[batch_idx].copy() for mat in self.data])

            epoch_num += 1

    def iterrows(self):
        """
        Iterate through indexes

        -------------------------------------------------------
        yield:
            index : int
                Index
        """
        for i in self.idx:
            yield self[i]

    def mask(self, mask):
        """
        Masking data matrices from DataFrame class

        -------------------------------------------------------
        Args:
            mask : List of n-dimensional data matrices
                The list of data matrices to return

        -------------------------------------------------------
        return:
            _ : DataFrame class
                DataFrame class with list of masked n-dimensional matrices
        """
        return DataFrame(copy.copy(self.columns), [mat[mask] for mat in self.data])

    def __iter__(self):
        return self.dict.items().__iter__()

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dict[key]

        elif isinstance(key, int):
            return pd.Series(dict(zip(self.columns, [mat[self.idx[key]] for mat in self.data])))

    def __setitem__(self, key, value):
        assert value.shape[0] == len(self), 'matrix first dimension does not match'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        self.dict[key] = value


class DataReader(object):
    """
    Class to read DataFrame class efficiently. Receives the directory with numpy files and column names and create
    DataFrame objects.

    Batch generator with 3 different modes implemented - train test and validation

    -------------------------------------------------------
    Args:
        data_dir : list
            Directory with numpy files to read

        data_col : list
            List with column names

        split : boolean
            With True then create 3 DataFrame objects. data_df, train_df, val_df, otherwise creates data_df

        train_size : int
            split_ratio number
    """
    def __init__(self, data_dir, data_col, split=True, train_size=0.9):

        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_col]

        self.df = DataFrame(columns = data_col, data = data)

        print('Data size', len(self.df))

        if split:
            self.train_df, self.val_df = self.df.train_test_split(train_size)
            print('Data train size', len(self.train_df))
            print('Data validation size', len(self.val_df))

    def batch_generator(self, batch_size, df, mode, shuffle=True, num_epochs=10000):
        """
        Batch_generator of DataReader object.
        It creates an interface to use batch generator of DataFrame class

        -------------------------------------------------------
        Args:
            batch_size : int
                Batch size to use

            df : DataFrame class
                Which dataframe to use

            mode : string
                train, test, val

            shuffle : boolean

            num_epochs : int

        -------------------------------------------------------
        yield:
            batch : DataFrame class
                Processed batch generated
        """
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode == 'test')
        )

        for batch in batch_gen:
            yield batch

    def train_batch_generator(self, batch_size):
        """
        Interface to create batch generator for train DataFrame object
        """
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            mode='train'
        )


    def val_batch_generator(self, batch_size):
        """
        Interface to create batch generator for validation DataFrame object
        """
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            mode='val'
        )

    def test_batch_generator(self, batch_size):
        """
        Interface to create batch generator for test DataFrame object
        """
        return self.batch_generator(
            batch_size=batch_size,
            df=self.df,
            shuffle=True,
            num_epochs=1,
            mode='test'
        )
