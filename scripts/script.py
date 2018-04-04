import warnings, os, argparse, sys

from skopt import gp_minimize
from skopt.space import Integer

warnings.filterwarnings("ignore")

from MyPackage.models import RNNTrainer

import torch

global run_number

torch.manual_seed(2)

def objective(params):
    try:
        model = get_model(params)
        scores = model.train_cv(args.folds, args.fold_size, args.patience)
        return scores

    except:
        return 10000.0


def get_model(params):
    x, y, z = params
    number_steps_train = int(x)
    hidden_size = int(y)
    num_layers = int(z)

    global run_number
    run_number = run_number + 1

    model = RNNTrainer(data_path='/datadrive/wind_power/wind_all.csv',
                       logger_path=path,
                       model_name='Run_number_' + str(run_number),
                       use_script=True,
                       lr=args.lr,
                       number_steps_train=number_steps_train,
                       number_steps_predict=args.predict_steps,
                       batch_size=args.batch_size,
                       num_epoch=args.epochs,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       cell_type=args.model,
                       target_column='SSG Wind',
                       validation_date='2015-01-01 00:00:00',
                       test_date='2016-01-01 00:00:00',
                       train_log_interval=args.train_log,
                       valid_log_interval=args.valid_log,
                       use_scheduler=args.scheduler,
                       normalizer=args.normalization,
                       optimizer=args.optimizer,
                       index_col=['Date and hour'],
                       parse_dates=True)

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script Variables')
    parser.add_argument('--file', default='runs', type=str, help='Directory to store files')
    parser.add_argument('--folds', default=3, type=int, help='Number of folds for cross val')
    parser.add_argument('--fold_size', default=365, type=int, help='Size in days for cross val fold')
    parser.add_argument('--predict_steps', type=int, help='Number of steps to forecast')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=10, type=int, help='Maximum number of epochs')
    parser.add_argument('--model', default='RNN', type=str, choices=['RNN', 'LSTM', 'GRU', 'QRNN', 'TCN', 'DRNN'],
                        help='Model to use. Check available models')
    parser.add_argument('--train_log', default=100, type=int, help='Number of steps to take log on train steps.')
    parser.add_argument('--valid_log', default=100, type=int, help='Number of steps to take log on valid steps.')
    parser.add_argument('--normalization', default='Standardization', type=str, help='Normalization to use',
                        choices=['Standardization', 'MixMaxScaler'])
    parser.add_argument('--scheduler', default=False, type=bool, help='Flag to choose to use lr scheduler')
    parser.add_argument('--train_steps', nargs=2, type=int, default=[10, 15], help='Interval to be optimized')
    parser.add_argument('--hidden_size', nargs=2, type=int, default=[5, 6], help='Interval to be optimized')
    parser.add_argument('--num_layers', nargs=2, type=int, default=[1, 2], help='Interval to be optimized')
    parser.add_argument('--initial_point', nargs=3, type=int, default=[10, 5, 2], help='Initial point for optimization')
    parser.add_argument('--N_CALLS', default=30, type=int, help='Number of calls for optmization')
    parser.add_argument('--RANDOM_STARTS', default=10, type=int, help='Number of random starts for optimization')
    parser.add_argument('--optimizer' ,default='Adam', type=str, choices=['Adam', 'SGD', 'RMSProp', 'Adadelta', 'Adagrad'],
                        help='Optimizer to use')
    parser.add_argument('--patience', default=3, type=int, help='Number of steps to stop train loop '
                                                                'after no improvment in validation set')

    parser.add_argument('--file_name', type=str, help='Name of file to save all the schnitzel')

    args = parser.parse_args()

    if not os.path.exists('/home/rneves/temp/temp_logger/' + args.file):
        path  = '/home/rneves/temp/temp_logger/' + args.file
        os.makedirs(path)
    else:
        sys.exit('This directory already exists. Check if you want to overwrite it. If so remove it manually.')

    SEED = 1337
    NCALLS = args.N_CALLS
    NRANDOMSTARTS = args.RANDOM_STARTS

    run_number = 0

    space  = [Integer(args.train_steps[0], args.train_steps[1]),# number_steps_train
              Integer(args.hidden_size[0], args.hidden_size[1]),# hidden_size
              Integer(args.num_layers[0], args.num_layers[1])]  # num_layers

    initial_point = args.initial_point

    res_gp = gp_minimize(objective,
                         space,
                         x0=initial_point,
                         n_calls=NCALLS,
                         random_state=SEED,
                         verbose=True,
                         n_random_starts=NRANDOMSTARTS)

    print(res_gp)

    print(res_gp.x)

    best_number_steps_train = res_gp.x[0]
    best_hidden_size = res_gp.x[1]
    best_num_layers = res_gp.x[2]

    model = RNNTrainer(data_path = '/datadrive/wind_power/wind_all.csv',
                       logger_path = path,
                       model_name = 'Run_Best_Model',
                       use_script = True,
                       lr = args.lr,
                       number_steps_train = best_number_steps_train,
                       number_steps_predict = args.predict_steps,
                       batch_size = args.batch_size,
                       num_epoch = args.epochs,
                       hidden_size = best_hidden_size,
                       num_layers = best_num_layers,
                       cell_type = args.model,
                       target_column = 'SSG Wind',
                       validation_date = '2015-01-01 00:00:00',
                       test_date = '2016-01-01 00:00:00',
                       train_log_interval = args.train_log,
                       valid_log_interval = args.valid_log,
                       use_scheduler = args.scheduler,
                       normalizer = args.normalization,
                       optimizer = args.optimizer,
                       index_col = ['Date and hour'],
                       parse_dates = True )

    model.train(args.patience)
    predictions, labels = model.predict()
    predictions, true_data, mse, mae = model.postprocess(predictions, labels)