# Wind Power Generation Forecasting

This is a experiment made along my master thesis with the principal focus of comparing different deep learning strategies applied on time series problems. This study was focused only in recurrent and convolutional architectures.

# Data

The data was collected by Redes Energéticas Nacionais (REN) and consists on the injected wind
power in the Portuguese power system. It was sampled at a 15 minutes resolution from the first day
of 2010 until the last day of 2016. The data collected pertains to all wind farms that are connected to
REN’s telemetry system. Data is under `data` folder.

![alt text](https://github.com/RodrigoNeves95/DeepLearningTimeSeries/blob/master/figures/WindPower.png "Wind Power Example")

# Objective

The main task is to make a forecasting of the wind power generated. Three horizons are going to be forecast. One, Six and 24 hours, meaning that for the one hour ahead predictions 4 points will be predicted (15, 30, 45 and 60 minutes).

# Algorithms

The following list of architectures were tested:

  * RNN Architectures [RNN + GRU + LSTM cell]
  * Dilated Recurrent Architectures [paper](https://arxiv.org/abs/1710.02224) [code](https://github.com/code-terminator/DilatedRNN)
  * Encoder-Decoder Architectures [paper](https://arxiv.org/abs/1406.1078)
  * Encoder-Decoder + Attention System Architectures [paper](https://arxiv.org/abs/1409.0473)

  * Quasi-RNN [paper](https://arxiv.org/abs/1611.01576) [code](https://github.com/salesforce/pytorch-qrnn)
  * Wavenet [paper](https://arxiv.org/abs/1609.03499v2) [code](https://gist.github.com/lirnli/4282fcdfb383bb160cacf41d8c783c70)
  * TCN [paper](https://arxiv.org/abs/1803.01271) [code](https://github.com/locuslab/TCN)
  
For a detailed explanation about the models check the links.
  
# Methodology

The data from the year of 2016 was used as test set. To train all the algorithms CV-sliding window technique is used. To optimize the hyperparameters it was used [skopt](https://scikit-optimize.github.io/) pakage. The data is standardized during model training (zero mean, unit variance). The models will use previous steps to predict the next ones (Autoregressive problem). The sequence length to train it will be optimized during the train.

# Requirements

This codebase was developed using Python 3 and PyTorch 0.3.1. Only GPU is supported for now.

Install the requiremnts `pip install -r requirements.txt`.

To run the scripts you should install the `mypackage`

```
git clone https://github.com/RodrigoNeves95/DeepLearningTimeSeries
cd DeepLearningTimeSeries
pip install -e mypackage
```

# Usage

## Recurrent Architecures + TCN and QRNN

This script will train the `model_to_run`, where the skopt package will optimize the hyperparameters. Then the best model will be used on the test set. The results are saved on `models_storage_folder/model_to_run/Best_Model_Predictions_[4, 24 or 96]/`

In this case the same model is used to predict the three different horizons. Check the argparse help for changng other variables.

```
python RNNScript.py --data_path path_to_data \
                    --SCRIPTS_FOLDER models_storage_folder \
                    --model model_to_run \
                    --file file_name
```

## Encoder-Decoder Architectures

This script will use the `model_to_run` to train and make predictions, where the skopt package will optimize the hyperparameters. Then the best model will be used on the test set. The results are saved on `models_storage_folder/model_to_run/Run_Best_Model_[4, 24 or 96]/`

```
python EncDecScript.py --data_path path_to_data \
                       --SCRIPTS_FOLDER models_storage_folder \
                       --model model_to_run \
                       --file file_name \
                       --use_attention True/False \
                       --predict_steps [4, 24 or 96]
```

## Wavenet

This script will use the `model_to_run` to train and make predictions, where the skopt package will optimize the hyperparameters. Then the best model will be used on the test set. The results are saved on `models_storage_folder/model_to_run/Run_Best_Model_[4, 24 or 96]/`

```
python WaveNetScript.py --data_path path_to_data \
                        --SCRIPTS_FOLDER models_storage_folder \
                        --model model_to_run \
                        --file file_name \
                        --predict_steps [4, 24 or 96]
```

