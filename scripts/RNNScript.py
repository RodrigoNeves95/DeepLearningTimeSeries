import warnings, os, argparse, sys

from skopt import gp_minimize
from skopt.space import Integer

warnings.filterwarnings("ignore")

from MyPackage.models import RNNTrainer

global run_number


def objective(params):
    try:
        model = get_model(params)
        scores = model.train_cv(args.folds, args.fold_size, args.patience)
        return scores

    except:
        return 10000.0


def get_model(params):
    x, y, z, k = params
    number_steps_train = int(x)
    hidden_size = int(y)
    num_layers = int(z)
    kernel_size = int(k)

    global run_number
    run_number = run_number + 1

    model = RNNTrainer(data_path=args.data_path,
                       logger_path=path,
                       model_name='Run_number_' + str(run_number),
                       lr=args.lr,
                       number_steps_train=number_steps_train,
                       number_steps_predict=args.predict_steps,
                       batch_size=args.batch_size,
                       num_epoch=args.epochs,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       kernel_size=kernel_size,
                       cell_type=args.model,
                       train_log_interval=args.train_log,
                       valid_log_interval=args.valid_log,
                       use_scheduler=args.scheduler,
                       normalizer=args.normalization,
                       optimizer=args.optimizer,
                       use_script=True,
                       target_column='Power',
                       validation_date='2015-01-01 00:00:00',
                       test_date='2016-01-01 00:00:00',
                       index_col=['Date'],
                       parse_dates=True)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script Variables')
    parser.add_argument('--SCRIPTS_FOLDER', default='/home/rneves/temp/temp_logger', type=str,
                        help='Main Folder to save all files')
    parser.add_argument('--data_path', default='/datadrive/wind_power/data/wind_15min.csv', type=str,
                        help='path for data file')
    parser.add_argument('--file', default='runs', type=str,
                        help='Directory to store files')
    parser.add_argument('--folds', default=3, type=int,
                        help='Number of folds for cross val')
    parser.add_argument('--fold_size', default=365, type=int,
                        help='Size in days for cross val fold')
    parser.add_argument('--predict_steps', type=int, default=1,
                        help='Number of steps to forecast')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='learning rate')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch Size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('--model', default='RNN', type=str,
                        choices=['RNN', 'LSTM', 'GRU', 'QRNN', 'TCN', 'DRNN'],
                        help='Model to use. Check available models')
    parser.add_argument('--train_log', default=100, type=int,
                        help='Number of steps to take log on train steps.')
    parser.add_argument('--valid_log', default=100, type=int,
                        help='Number of steps to take log on valid steps.')
    parser.add_argument('--normalization', default='Standardization', type=str,
                        choices=['Standardization', 'MixMaxScaler'],
                        help='Normalization to use')
    parser.add_argument('--scheduler', default=False, type=bool,
                        help='Flag to choose to use lr scheduler')
    parser.add_argument('--train_steps', nargs=2, type=int, default=[50, 1000],
                        help='Interval to be optimized')
    parser.add_argument('--hidden_size', nargs=2, type=int, default=[5, 60],
                        help='Interval to be optimized')
    parser.add_argument('--num_layers', nargs=2, type=int, default=[1, 10],
                        help='Interval to be optimized')
    parser.add_argument('--kernel_size', nargs=2, type=int, default=[2, 50],
                        help='Interval of kernel size for TCN and QRNN models')
    parser.add_argument('--initial_point', nargs=4, type=int, default=[50, 10, 2, 10],
                        help='Initial point for optimization')
    parser.add_argument('--N_CALLS', default=40, type=int,
                        help='Number of calls for optmization')
    parser.add_argument('--RANDOM_STARTS', default=10, type=int,
                        help='Number of random starts for optimization')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        choices=['Adam', 'SGD', 'RMSProp', 'Adadelta', 'Adagrad'],
                        help='Optimizer to use')
    parser.add_argument('--patience', default=3, type=int,
                        help='Number of steps to stop train loop after no improvment in validation set')
    parser.add_argument('--steps_to_predict', type=int, default=[4, 24, 96], nargs=3,
                        help='Steps for predict using best model after optimization')


    args = parser.parse_args()

    if not os.path.exists(args.SCRIPTS_FOLDER + args.file):
        path = args.SCRIPTS_FOLDER + '/' + args.file
        os.makedirs(path)
    else:
        sys.exit('This directory already exists. Check if you want to overwrite it, then remove it manually.')

    SEED = 1337

    NCALLS = args.N_CALLS
    NRANDOMSTARTS = args.RANDOM_STARTS

    run_number = 0

    if args.model in ['TCN', 'QRNN']:
        space = [Integer(args.train_steps[0], args.train_steps[1]),  # number_steps_train
                 Integer(args.hidden_size[0], args.hidden_size[1]),  # hidden_size
                 Integer(args.num_layers[0], args.num_layers[1]),    # num_layers
                 Integer(args.kernel_size[0], args.kernel_size[1])]  # kernel_size
        initial_point = args.initial_point
    elif args.model in ['RNN', 'LSTM', 'GRU', 'DRNN'] :
        space = [Integer(args.train_steps[0], args.train_steps[1]),  # number_steps_train
                 Integer(args.hidden_size[0], args.hidden_size[1]),  # hidden_size
                 Integer(args.num_layers[0], args.num_layers[1]),    # num_layers
                 Integer(10, 10)]                                    # kernel_size
        initial_point = args.initial_point

    res_gp = gp_minimize(objective,
                         space,
                         x0=initial_point,
                         n_calls=NCALLS,
                         random_state=SEED,
                         verbose=True,
                         n_random_starts=NRANDOMSTARTS)

    best_number_steps_train = int(res_gp.x[0])
    best_hidden_size = int(res_gp.x[1])
    best_num_layers = int(res_gp.x[2])

    if args.model in ['TCN', 'QRNN']:
        best_kernel_size = int(res_gp.x[3])
    else:
        best_kernel_size = 10

    model = RNNTrainer(data_path=args.data_path,
                       logger_path=path,
                       model_name='Run_Best_Model',
                       lr=args.lr,
                       number_steps_train=best_number_steps_train,
                       number_steps_predict=args.predict_steps,
                       batch_size=args.batch_size,
                       num_epoch=args.epochs,
                       hidden_size=best_hidden_size,
                       num_layers=best_num_layers,
                       kernel_size=best_kernel_size,
                       cell_type=args.model,
                       train_log_interval=args.train_log,
                       valid_log_interval=args.valid_log,
                       use_scheduler=args.scheduler,
                       normalizer=args.normalization,
                       optimizer=args.optimizer,
                       use_script=True,
                       target_column='Power',
                       validation_date='2015-01-01 00:00:00',
                       test_date='2016-01-01 00:00:00',
                       index_col=['Date'],
                       parse_dates=True)

    model.train(args.patience)

    for range in args.steps_to_predict:

        model = RNNTrainer(data_path=args.data_path,
                           logger_path=path,
                           model_name='Best_Model_Predictions_' + str(range),
                           lr=args.lr,
                           number_steps_train=best_number_steps_train,
                           number_steps_predict=range,
                           batch_size=args.batch_size,
                           num_epoch=args.epochs,
                           hidden_size=best_hidden_size,
                           num_layers=best_num_layers,
                           kernel_size=best_kernel_size,
                           cell_type=args.model,
                           train_log_interval=args.train_log,
                           valid_log_interval=args.valid_log,
                           use_scheduler=args.scheduler,
                           normalizer=args.normalization,
                           optimizer=args.optimizer,
                           use_script=True,
                           target_column='Power',
                           validation_date='2015-01-01 00:00:00',
                           test_date='2016-01-01 00:00:00',
                           index_col=['Date'],
                           parse_dates=True)

        dir = path + '/Run_Best_Model'
        model.get_best(dir)
        model.model.number_steps_predict = range
        predictions, labels = model.predict()
        final_df, mse, mae = model.postprocess(predictions, labels)
        model.filelogger.write_results(predictions, labels, final_df, mse, mae)
