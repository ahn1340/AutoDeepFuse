from __future__ import division
import torch
from torch import nn
from model_zoo import resnext,resnet,resnet2p1d
import pdb
from collections import OrderedDict

def generate_model(opt, model_arch,num_classes,pretrain_path):
    #assert opt.model in ['resnext','resnet','resnet2p1d']
    #assert opt.model_depth in [101]
    from model_zoo.resnext import get_fine_tuning_parameters

    if model_arch == 'resnet18':
        model = resnet.generate_model(model_depth=18,
                                      n_classes=num_classes,
                                      n_input_channels=opt.input_channels)
                                      #shortcut_type=opt.resnet_shortcut)
        setattr(model, 'num_channels', 256)

    elif model_arch == 'resnet34':
        model = resnet.generate_model(model_depth=34,
                                      n_classes=num_classes,
                                      n_input_channels=opt.input_channels)
                                      #shortcut_type=opt.resnet_shortcut)          
    elif model_arch == 'resnet50':
        model = resnet.generate_model(model_depth=50,
                                      n_classes=num_classes,
                                      n_input_channels=opt.input_channels)
                                      #shortcut_type=opt.resnet_shortcut)  
        setattr(model, 'num_channels', 1024)
    elif model_arch == 'resnet152':
        model = resnet.generate_model(model_depth=152,
                                      n_classes=num_classes,
                                      n_input_channels=opt.input_channels)
                                      #shortcut_type=opt.resnet_shortcut)
        setattr(model, 'num_channels', 1024)
    elif model_arch == 'resnet200':
        model = resnet.generate_model(model_depth=200,
                                      n_classes=num_classes,
                                      n_input_channels=opt.input_channels)
                                      #shortcut_type=opt.resnet_shortcut)  
        setattr(model, 'num_channels', 1024)

    elif model_arch == 'resnet2p1d18':
        model = resnet2p1d.generate_model(model_depth=18,
                                          n_classes=num_classes,
                                          n_input_channels=opt.input_channels)
                                          #shortcut_type=opt.resnet_shortcut)
    
    elif model_arch == 'resnet2p1d34':
        model = resnet2p1d.generate_model(model_depth=34,
                                          n_classes=num_classes,
                                          n_input_channels=opt.input_channels)
                                          #shortcut_type=opt.resnet_shortcut)
        
    elif model_arch == 'resnet2p1d50':
        model = resnet2p1d.generate_model(model_depth=50,
                                          n_classes=num_classes,
                                          n_input_channels=opt.input_channels)
                                          #shortcut_type=opt.resnet_shortcut)
        
        setattr(model, 'num_channels', 1024)

    elif model_arch == 'resnext101':
        
        model = resnext.resnet101(
                num_classes=num_classes,
                #shortcut_type=opt.resnet_shortcut,
                #cardinality=opt.resnext_cardinality,
                sample_size=112,
                sample_duration=opt.training.num_segments,
                input_channels=opt.input_channels)
                #output_layers=opt.output_layers)
        setattr(model, 'num_channels', 1024)
    else:
        raise Exception('Unknown architecture')
    
    if pretrain_path:
        #assert opt.arch == pretrain['arch']
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        #print(pretrain)
        if 'resnext' in model_arch:
            #model = model.cuda()
            #model = nn.DataParallel(model)
            # Create a new state dict to store keys without "module." prefix.
            # necessary to load weights trained with DataParallel
            new_state_dict = OrderedDict()
            for k, v in pretrain['state_dict'].items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            #model.load_state_dict(pretrain['state_dict'])

        else:
            model.load_state_dict(pretrain['state_dict'])
            #model = model.cuda()

        model.fc = nn.Linear(model.fc.in_features, opt.datasets.num_class)
        #model.fc = model.fc.cuda()

    # Freeze the paramters of the pretrained models
    for param in model.parameters():
        param.requires_grad = False

    return model#, model.parameters()

