import os
import json

from utils.dirs import create_dirs
from utils.config import process_config
from utils.utils import get_args, get_logger

from cov_model import COVsearchModel
from data_loader import get_train_val_test_dataloader

from models import select_model
from torch import cuda
from arch_search.create_random_arch import create_random_architecture


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

    if args.genotype is not None:  # Genotype is obtained from search
        arch = eval(json.load(open(os.path.join(config.log_dir,
                                                args.genotype))))
        #input_gene = eval(json.load(open(os.path.join(config.log_dir,
        #                                              args.input_genotype))))
        print("#"*30)
        print("Genotype: {}".format(arch))
        print("#"*30)

    elif args.random:  # random architecture with uniform sampling
        num_cells = config.network.arch_search.cells
        arch = create_random_architecture(num_cells)
        # Store genotype in json file.
        json.dump(str(arch), open(os.path.join(config.log_dir, 'genotype_all.json'), 'w'))

        print("#"*30)
        print("Genotype: {}".format(arch))
        print("#"*30)

    else:  # Baseline
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
                       discretization='all',
                       )

    #print("line"*30)
    model = COVsearchModel(net, config, logger)

    train_loader, val_loader, test_loader = get_train_val_test_dataloader(config, net)
    #test_loader = get_test_dataloader(config.datasets.test)


    print("mline1")

    if config.mode == 'train':
        model.train(train_loader, val_loader, test_loader)
    #elif config.mode == 'test':
        #model.test(test_loader)

if __name__ == '__main__':
    main()
