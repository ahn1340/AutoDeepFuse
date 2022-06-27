import torch
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from MTLULoss import MTLULoss

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def select_optimizer(cfg, params):
    if cfg.name == 'sgd':
        print(" %%%%%% Using SGD optimizer %%%%%%")
        optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    elif cfg.name == 'rms':
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
        optimizer = torch.optim.RMSprop(params, lr=cfg.lr)
    elif cfg.name == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=cfg.weight_decay,
        eps=0.001)
    elif cfg.name == 'adamw':
        print(" %%%%%% Using ADAMW optimizer %%%%%%")
        from optimizers.adamw import AdamW
        optimizer = AdamW(params, weight_decay=cfg.weight_decay,
        eps=0.001)
    elif cfg.name == 'radam':
        print(" %%%%%% Using RADAM optimizer %%%%%%")
        from optimizers.radam import RAdam
        optimizer = RAdam(params, weight_decay=cfg.weight_decay,
        eps=0.001)
    else:
        raise Exception('Unknown Optimizer')
    return optimizer


def get_params_opt(cfg, model):
    params_dict = dict(model.named_parameters())
    params = []
    #print(cfg.training.representation)
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0
        #print("key: ", key)
        if ('.base.conv1' in key
                or '.base.bn1' in key
                or 'data_bn' in key):
            lr_mult = 0.1
            #print("here==="*20)
        elif '.fc.' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.01
        #print("lr_mult: ", lr_mult)
        if value.requires_grad is True:
            params += [{'params': value, 'lr': cfg.training.optimizer.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]
        
    return params
    
def get_criterion(cfg):
    if cfg.training.loss_func == "CrossEntropy":
       criterion = torch.nn.CrossEntropyLoss().cuda()
    elif cfg.training.loss_func == "CrossEntropyUW":
       criterion = MTLULoss().cuda()     
         
    return criterion


