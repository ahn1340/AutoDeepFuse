import os
import json

from utils.dirs import create_dirs
from utils.config import process_config
from utils.utils import get_args, get_logger

from cov_model import COVsearchModel
from data_loader import get_train_val_test_dataloader

from models import select_model
from torch import cuda


def main():
    args = get_args()
    config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config.logging.model_dir_pre,
                 config.logging.log_dir_pre,
                 config.log_dir,
                 config.discretized_model_dir,
                 ])
    # logging to the file and stdout
    logger = get_logger(config.log_dir, config.exp_name)

    #print("config.network.arch: ", config.network.arch)
    #net = Model(config.datasets.num_class, config.training.num_segments, config.training.representation,
    #              base_model=config.network.arch)

    if args.genotype is not None:
        arch = eval(json.load(open(os.path.join(config.log_dir,
                                                args.genotype))))
        #input_gene = eval(json.load(open(os.path.join(config.log_dir,
        #                                              args.input_genotype))))
        print("#"*30)
        print("Genotype: {}".format(arch))
        #print("Input Genotype: {}".format(input_gene))
        print("#"*30)
    else:
        arch = None
        print("#"*30)
        print("No Genotype given.")
        print("#"*30)


    net = select_model(config,
                       config.datasets.num_class,
                       config.training.num_segments,
                       config.network,
                       dropout=config.training.dropout,
                       genotype=arch,
                       truncated_pretrained=True
                       )

    #print("line"*30)
    model = COVsearchModel(net, config, logger)

    train_val_loader, test_loader = get_train_val_test_dataloader(config, net)
    #test_loader = get_test_dataloader(config.datasets.test)


    print("mline1")

    if config.mode == 'train':
        model.train(train_val_loader, test_loader)
    #elif config.mode == 'test':
    #model.test(test_loader)

if __name__ == '__main__':
    main()
