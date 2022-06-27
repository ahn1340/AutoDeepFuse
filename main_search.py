from torch.nn import CrossEntropyLoss
from utils.dirs import create_dirs
from utils.config import process_config
from utils.utils import get_args, get_logger

from arch_search.cells import Network
from arch_search.train_search import COVsearchModel
from data_loader import get_train_val_dataloader
from torch import cuda


def main():
    print("Cuda availability: {}".format(cuda.is_available()))
    args = get_args()
    config = process_config(args.config)


    # create the experiments dirs
    create_dirs([config.logging.model_dir_pre, config.logging.log_dir_pre, config.log_dir,config.model_dir])
    #print(config.logging.model_dir_pre)
    #print(config.model_dir)
    print("log_dir:", config.log_dir)
    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.exp_name)

    print("config.network.arch: ", config.network.arch)
    #net = Model(config.datasets.num_class, config.training.num_segments, config.training.representation,
    #              base_model=config.network.arch)


    net = Network(config,
                  config.network,
                  config.datasets.num_class,
                  config.training.num_segments,
                  CrossEntropyLoss,
                  dropout=config.training.dropout)

    #print("line"*30)
    model = COVsearchModel(net, config, logger)

    train_loader, val_loader = get_train_val_dataloader(config, net)
    #test_loader = get_test_dataloader(config.datasets.test)


    print("mline1")

    if config.mode == 'train':
        model.train(train_loader, val_loader)
    #elif config.mode == 'test':
        #model.test(test_loader)

if __name__ == '__main__':
    main()
