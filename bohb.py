import os
import sys
import traceback
import math
import numpy as np
import argparse
import time
import pickle

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# hpbandster
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import random

# genotype
import json
from utils.config import process_config
from utils.utils import get_logger
from utils.dirs import create_dirs
from collections import namedtuple
Genotype = namedtuple(
    'Genotype',
    'first first_concat'
)

# model
from models import select_model
from cov_model_bohb import COVsearchModel
from data_loader import get_train_val_test_dataloader


def get_configspace():
    '''
    Hyperparamters: optimizer (learning rate), weight decay, channel size, batch size
    '''
    cs = CS.ConfigurationSpace()
    # Model-independent hyperparamters
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='batch_size', choices = [4, 8, 16, 32]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-5, upper=1e-2, log=True))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='optimizer', choices=['adam', 'adamw', 'sgd']))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='weight_decay', lower=1e-5, upper=2e-2, log=True))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='num_channels', lower=128, upper=1024, log=True))

    return cs


def get_configuration(genotype, model_config):
    cfg = {}
    cfg['genotype_path'] = genotype  # path to the genotype.json file
    cfg['yaml_config_path'] = model_config  # path to the yaml file of the configuration
    cfg["bohb_min_budget"] = 2  # epoch
    cfg["bohb_max_budget"] = 70  # epoch
    cfg["bohb_iterations"] = 10
    cfg["bohb_eta"] = 3
    cfg["bohb_log_dir"] = "./bohb_log/"

    #TODO: I may need to add the path to genotype here
    return cfg


class BOHBWorker(Worker):
    def __init__(self, cfg, logger, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.cfg = cfg
        self.logger = logger

    def compute(self, config, budget, *args, **kwargs):
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        budget = math.floor(budget)  # round down
        print('BUDGET: ' + str(budget))

        info = {}

        score = 0 
        try:
            print('BOHB ON DATASET: ' + str('HMDB51'), file=sys.stderr)
            score = execute_run(cfg=cfg, config=config, budget=budget, logger=logger)
            print("exec run done")
        except Exception:
            status = traceback.format_exc()
            print("STATUS: ", status, file=sys.stderr)
            score = -np.inf  # if a run crashes (due to memory error for example), give it the worst possible score

        info['accuracy'] = score
        info['config'] = str(config)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('----------------------------')
        print("END BOHB ITERATION")

        return {
            "loss": -score,
            "info": info
        }


def execute_run(cfg, config, budget, logger):  # config: configuration space config
    # Get genotype and create model
    arch = eval(json.load(open(cfg['genotype_path'])))
    print("#"*30)
    print("Genotype: {}".format(arch))
    print("#"*30)

    model_config = cfg['yaml_config_path']

    # create the experiments dirs
    create_dirs([model_config.logging.model_dir_pre,
                 model_config.logging.log_dir_pre,
                 model_config.log_dir,
                 model_config.discretized_model_dir,
                 ])

    # set sampled configuration (num_channels, optimizer, learning rate, weight_decay)
    model_config.training.optimizer.name = config['optimizer']
    model_config.training.optimizer.lr = config['lr']
    model_config.training.optimizer.weight_decay = config['weight_decay']
    model_config.training.batch_size = config['batch_size']
    #model_config.network.arch_search.channels = config['num_channels']

    # assign budget (epoch)
    model_config.training.num_epochs_train = budget

    net = select_model(model_config,
                       model_config.datasets.num_class,
                       model_config.training.num_segments,
                       model_config.network,
                       dropout=0,  # no dropout in our experiments
                       genotype=arch,
                       )

    model = COVsearchModel(net, model_config, logger)

    train_loader, val_loader, test_loader = get_train_val_test_dataloader(model_config, net)

    # start training and get the final test accuracy
    test_accuracy = model.train(train_loader, val_loader, test_loader)

    return test_accuracy

def runBOHB(cfg, logger, shared_directory, host, result_folder, run_id):
    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    if args.worker:
        time.sleep(5)  # short artificial delay to make sure the nameserver is already running
        w = BOHBWorker(cfg=cfg, logger=logger, run_id=run_id, host=host)
        w.load_nameserver_credentials(working_directory=shared_directory)
        w.run(background=False)
        exit(0)

    ns = hpns.NameServer(run_id=run_id, host=host, port=port, working_directory=shared_directory)
    ns_host, ns_port = ns.start()
    print("HOST: {}".format(host))
    print("NS_HOST: {}, NS_PORT: {}".format(ns_host, ns_port))

    w = BOHBWorker(cfg=cfg, logger=logger, host=host, nameserver=ns_host, run_id=run_id, nameserver_port=ns_port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(
        directory=os.path.join(cfg["bohb_log_dir"], result_folder), overwrite=True
    )

    bohb = BOHB(
        configspace=get_configspace(),
        run_id=run_id,
        min_budget=cfg["bohb_min_budget"],
        max_budget=cfg["bohb_max_budget"],
        eta=cfg["bohb_eta"],
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
    )

    res = bohb.run(n_iterations=cfg["bohb_iterations"])

    # Store results
    with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for running BOHB on cluster')
    parser.add_argument('--genotype', type=str, help='path to genotype')
    parser.add_argument('--yaml_config', type=str, help='path to yaml_config')
    parser.add_argument('--result_folder', type=str, help='folder to store configs and results')
    parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=2)
    parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=70)
    parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=10)
    parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.')
    parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=8)
    parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    parser.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')
    args = parser.parse_args()
    shared_directory = args.shared_directory
    result_folder = args.result_folder

    host = hpns.nic_name_to_host(args.nic_name)

    # logging to the file and stdout
    model_config = process_config(args.yaml_config)

    model_config.logging.model_dir_pre ='bohb_' + model_config.logging.model_dir_pre
    model_config.logging.log_dir_pre ='bohb_' + model_config.logging.log_dir_pre

    model_config.save_result_name = model_config.network.arch_rgb + "_"  + model_config.network.arch_flow + "_" + model_config.network.arch_pose + "_Batch:" \
                                    + str(model_config.training.batch_size) + "_Optimizer:" + model_config.training.optimizer.name \
                                    + "_Seg"  + str(model_config.training.num_segments) + "_" + model_config.exp_name
    #print(config.save_result_name)
    model_config.log_dir = os.path.join(model_config.logging.log_dir_pre, model_config.save_result_name)
    model_config.model_dir = os.path.join(model_config.log_dir, 'models')
    model_config.discretized_model_dir = os.path.join(model_config.log_dir, 'discretized_models')


    create_dirs([model_config.logging.model_dir_pre,
                 model_config.logging.log_dir_pre,
                 model_config.log_dir,
                 model_config.discretized_model_dir,
                 ])

    logger = get_logger(model_config.log_dir, model_config.exp_name) # hmdb_c1_run1

    #if len(sys.argv) == 3:      # parallel processing
    #    for arg in sys.argv[1:]:
    #        print(arg)

    #    id = int(sys.argv[1])
    #    tot = int(sys.argv[2])
    #    for i, dataset in enumerate(datasets):
    #        if (i-id)%tot != 0:
    #            continue
    #        for model in models:
    #            cfg = get_configuration(dataset, genotype, yaml_config)
    #            res = runBOHB(cfg)
    cfg = get_configuration(args.genotype, model_config)
    res = runBOHB(cfg, logger, shared_directory, host, result_folder, run_id=args.run_id)


    # print result
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
