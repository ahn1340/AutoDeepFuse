import os
import time
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
import GPUtil
import shutil
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.utils import clip_grad_norm_

# custom modules
from utils.metrics import AverageMeter
from utils.utils import to_device
from schedulers import select_scheduler
from optimizers import select_optimizer, get_params_opt, get_criterion
import random

import json
import glob

#import swats

# summary
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


class COVsearchModel:
    def __init__(self, model, args, logger):
        self.args = args
        self.logger = logger

        json.dump(self.args, open(os.path.join(self.args.log_dir, 'kwargs.json'), 'w'))
        train_writer_folder = os.path.join(self.args.log_dir, "train_log")
        try:
            os.mkdir(train_writer_folder)
            print("Directory", train_writer_folder, "Created")
        except FileExistsError:
            print("Directory", train_writer_folder, "already exists.")
        self.writer = SummaryWriter(log_dir=train_writer_folder)

        cudnn.enabled = True

        # set up model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.start_epoch = 0
        self.step_valid_loss = 10

        # For storing final performance
        self.val_acc1_list = []
        self.val_acc5_list = []
        self.test_acc1_list = []
        self.test_acc5_list = []
        self.val_best = None
        self.test_best = 0
        self.val_best_epoch = 0

        # self.model = self.model.to(self.device)
        # print("Optimizer: ", optimizer)
        self.model = torch.nn.DataParallel(self.model).cuda()
        #self.model = self.model.cuda()  # no multi GPU training for BOHB

        # fix random seed to reproduce results
        if args.random_seed is not None:
            random.seed(args.random_seed)
            torch.manual_seed(args.random_seed)
            cudnn.deterministic = True
            logger.info('Random seed: {:d}'.format(args.random_seed))

        # ---- Select optimizer and scheduler ---- #
        params = get_params_opt(args, self.model)
        self.loss_func = get_criterion(args)

        # For training the found architecture, we fix the weight decay as default = 0.0003
        args.training.optimizer.weight_decay = 0.0003
        self.optimizer = select_optimizer(args.training.optimizer, params)

        self.scheduler = select_scheduler(self.optimizer, args.training.lr_scheduler, train_or_search="train")

        if args.mode == 'train':

            self.start_iter = 0

            # resume
            list_of_files = glob.glob(self.args.discretized_model_dir + '/*')  # * means all if need specific format then *.csv

            if list_of_files:
                print("--"*30)
                print("Resume training ....")
                latest_checkpoint = max(list_of_files, key=os.path.getctime)
                self.load(latest_checkpoint)
                cudnn.benchmark = True
            else:
                if self.args.training.finetune:
                    print("no finetuning")
                    print("--"*30)
                else:
                    print("--"*30)
                    print("Start training with default pretrained (ImageNet)...")
                    lr = self.args.training.optimizer.lr
                    #wd = self.args.training.optimizer.weight_decay
                    wd = 0.0003  # Fix the weight decay to the baseline when training
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr #* param_group['lr_mult']
                        param_group['weight_decay'] = wd #* param_group['decay_mult']
                    print("--"*30)

            cudnn.benchmark = True

        elif args.mode == 'val':
            self.load(os.path.join(args.model_dir, args.validation.model))
        else:
            self.load(os.path.join(args.model_dir, args.testing.model))



    def train(self, train_loader, val_loader, test_loader):
        self.best_prec1 = 0

        max_step = len(train_loader)
        print('\n[+] Start training')

        #if torch.cuda.device_count() > 1:
        #    print('\n[+] Use {} GPUs'.format(torch.cuda.device_count()))
        #    self.model = nn.DataParallel(self.model)  # ,device_ids=[1])

        start_t_el = time.time()
        #prefetcher = data_prefetcher(train_loader,num_segments=3)
        #valid_loss = 10

        for epoch in range(self.start_epoch, self.args.training.num_epochs_train):
            self.model.train()
            # In PyTorch 0.4, "volatile=True" is deprecated.
            torch.set_grad_enabled(True)

            ################################################################
            start_t2 = time.time()

            # set model drop path probability
            self.model.drop_path_prob = self.args.training.dropout * (epoch / self.args.training.num_epochs_train)
            print("Epoch {} drop_path probability: {}".format(epoch, self.model.drop_path_prob))

            loss_summ = 0
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            data_time = AverageMeter()

            iter_count = self.args.training.itersize
            print('====> EffectiveBatchSize: ', iter_count * self.args.training.batch_size)
            for step, (input_rgb, input_flow, input_pose, target) in enumerate(train_loader):
               #input, target = prefetcher.next()
                gl_step = epoch * max_step + step  # counter for tensorboard
                data_time.update((time.time() - start_t2)/60)

                if self.args.use_cuda:
                    target = target.cuda(non_blocking=True)
                    input_rgb = input_rgb.cuda(non_blocking=True)
                    input_flow = input_flow.cuda(non_blocking=True)
                    input_pose = input_pose.cuda(non_blocking=True)

                input_rgb_var = torch.autograd.Variable(input_rgb)
                input_flow_var = torch.autograd.Variable(input_flow)
                input_pose_var = torch.autograd.Variable(input_pose)

                target_var = torch.autograd.Variable(target)
                # compute output
                start_t = time.time()
                #output,out_rgb,out_flow,out_pose = self.model(input_rgb_var, input_flow_var, input_pose_var)
                output = self.model(input_rgb_var, input_flow_var, input_pose_var)
                forward_t = time.time() - start_t

                #print('#'*20)
                #print('OUTPUT: ', output.shape)
                #print('RGB: ', out_rgb.shape)
                #print('flow: ', out_flow.shape)
                #print('pose: ', out_pose.shape)
                #print('#'*20)

                loss = self.loss_func(output, target_var)
                #loss_rgb = self.loss_func(out_rgb, target_var)
                #loss_flow = self.loss_func(out_flow, target_var)
                #loss_pose = self.loss_func(out_pose, target_var)
                #loss = loss + loss_rgb + loss_flow +  loss_pose
                loss = loss / self.args.training.itersize
                loss_summ += loss
                    # print(loss, loss.shape)
                prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))
                losses.update(loss_summ.item(), input_rgb.size(0))
                top1.update(prec1.item(), input_rgb.size(0))
                top5.update(prec5.item(), input_rgb.size(0))
                iter_count -= 1

                start_t = time.time()

                loss.backward()

                backward_t = time.time() - start_t

                if self.args.training.clip_gradient is not None:
                    total_norm = clip_grad_norm_(self.model.parameters(), self.args.training.clip_gradient)
                    if total_norm > self.args.training.clip_gradient:
                        print(
                            "clipping gradient: {} with coef {}".format(total_norm,
                                                                        self.args.training.clip_gradient / total_norm))

               # Effective batch_size = itersize * batch_size
                if iter_count == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    iter_count = self.args.training.itersize
                    #print(step)
                    loss_summ = 0

                # Show GPU usage information
                if (step + 1) / self.args.training.itersize  == 1:
                      GPUtil.showUtilization(all=True)

                if (step + 1) % self.args.logging.print_step == 0:
                       print_string = 'Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t El_time {el_time:.3f} ({el_time:.3f})\t data_time {data_time.val:.3f} \t Loss ' \
                                      '{loss.val:.4f} ({loss.avg:.4f})\t Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                       self.logger.info(print_string.format(epoch, step, max_step,
                                                            el_time=(time.time() - start_t_el) / 60,
                                                            data_time=data_time ,
                                                            loss=losses,
                                                            top1=top1,
                                                            top5=top5,
                                                            lr=self.optimizer.param_groups[0]['lr']))

                       self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'],
                                              global_step=gl_step)
                       self.writer.add_scalar('train/acc1', top1.avg, global_step=gl_step)
                       self.writer.add_scalar('train/acc5', top5.avg, global_step=gl_step)
                       self.writer.add_scalar('train/loss', loss, global_step=gl_step)
                       self.writer.add_scalar('train/forward_time', forward_t, global_step=gl_step)
                       self.writer.add_scalar('train/backward_time', backward_t, global_step=gl_step)

            ################################################################
            # update learning rate
            self.scheduler.step(epoch)

            # Compute validation and test performance
            #if step % self.args.training.val_step == 0 or epoch == self.args.training.num_epochs - 1:
            _valid_res = self.validate(self.args, self.model, self.loss_func, val_loader, self.writer)
            _test_res = self.test(self.args, self.model, self.loss_func, test_loader, self.writer)
            # print('\n[+] Valid results')

            self.step_valid_loss = _valid_res[2]
            self.step_test_loss = _test_res[2]
            is_best = _valid_res[0] > self.best_prec1  # True if current checkpoint performance is bettwe
            self.best_prec1 = max(_valid_res[0], self.best_prec1)
            if is_best or epoch % self.args.save_chp_freq == 0:
                #self.save_checkpoint(self.args.model_dir, epoch, self.best_prec1, is_best, self.step_valid_loss)
                #print('\n[+] Model saved')
                pass  # Don't save model for now

            if self.val_best is None or _valid_res[0] > self.val_best:
                self.val_best = _valid_res[0]
                self.test_best = _test_res[0]
                self.val_best_epoch = epoch

            self.writer.add_scalar('test/acc1', _test_res[0], global_step=gl_step)
            self.writer.add_scalar('test/acc5', _test_res[1], global_step=gl_step)
            self.writer.add_scalar('test/loss', _test_res[2], global_step=gl_step)

            self.writer.add_scalar('valid/acc1', _valid_res[0], global_step=gl_step)
            self.writer.add_scalar('valid/acc5', _valid_res[1], global_step=gl_step)
            self.writer.add_scalar('valid/loss', _valid_res[2], global_step=gl_step)

            self.test_acc1_list.append(_test_res[0])
            self.test_acc5_list.append(_test_res[1])

            self.val_acc1_list.append(_valid_res[0])
            self.val_acc5_list.append(_valid_res[1])

            # write final accuracies to json file
            print("Saving performance dict..")
            performance_dict = {"val_acc1": self.val_acc1_list,
                                "val_acc5": self.val_acc5_list,
                                "test_acc1": self.test_acc1_list,
                                "test_acc5": self.test_acc5_list,
                                'val_best': self.val_best,
                                'test_best': self.test_best,
                                'val_best_epoch': self.val_best_epoch,
                                }
            json.dump(performance_dict, open(os.path.join(self.args.log_dir, 'performance_dict.json'), 'w'))
            print("Saved.")

        self.writer.close()

        return self.test_best

    def validate(self, args, model, criterion, val_loader, writer, device=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        infer_t = 0

        # In PyTorch 0.4, "volatile=True" is deprecated.
        torch.set_grad_enabled(False)
        model.eval()
        with torch.no_grad():
            end = time.time()
            for step, (input_rgb, input_flow, input_pose, target) in enumerate(val_loader):
                start_t = time.time()

                if self.args.use_cuda:
                    target = target.cuda(non_blocking=True)
                    input_rgb = input_rgb.cuda(non_blocking=True)
                    input_flow = input_flow.cuda(non_blocking=True)
                    input_pose = input_pose.cuda(non_blocking=True)

                input_rgb_var = torch.autograd.Variable(input_rgb, requires_grad=False)
                input_flow_var = torch.autograd.Variable(input_flow, requires_grad=False)
                input_pose_var = torch.autograd.Variable(input_pose, requires_grad=False)

                target_var = torch.autograd.Variable(target, requires_grad=False)

                output = model(input_rgb_var, input_flow_var, input_pose_var)
                loss = criterion(output, target_var)

                prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))
                #prec1_rgb, _ = self.accuracy(out_rgb.data, target, topk=(1, 5))
                #prec1_flow, _ = self.accuracy(out_flow.data, target, topk=(1, 5))
                #prec1_pose, _ = self.accuracy(out_pose.data, target, topk=(1, 5))

                losses.update(loss.data, input_rgb.size(0))
                top1.update(prec1.item(), input_rgb.size(0))
                top5.update(prec5.item(), input_rgb.size(0))

                batch_time.update(time.time() - end)
                infer_t += time.time() - start_t
                end = time.time()

                if step % args.logging.print_step == 0:
                    #print("RGB {:.2f} flow {:.2f} Residual {:.2f}".format(prec1_rgb.item(),prec1_flow.item(),prec1_ru.item()))
                    print(('Validation: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        step, len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5)))

        self.logger.info(
            'Validation Results: Prec@1 {:.2f} %, Prec@5  {:.2f} %, Loss  {:.4f} '.format(top1.avg, top5.avg, losses.avg))
        # if writer:
        #    n_imgs = min(images.size(0), 10)
        #    for j in range(n_imgs):
        #        writer.add_image('valid/input_image',
        #                         concat_image_features(images[j], first[j]), global_step=step)
        return top1.avg, top5.avg, losses.avg, infer_t

    def test(self, args, model, criterion, test_loader, writer, device=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        infer_t = 0

        # In PyTorch 0.4, "volatile=True" is deprecated.
        torch.set_grad_enabled(False)
        model.eval()
        with torch.no_grad():
            end = time.time()
            for step, (input_rgb, input_flow, input_pose, target) in enumerate(test_loader):
                start_t = time.time()

                if self.args.use_cuda:
                    target = target.cuda(non_blocking=True)
                    input_rgb = input_rgb.cuda(non_blocking=True)
                    input_flow = input_flow.cuda(non_blocking=True)
                    input_pose = input_pose.cuda(non_blocking=True)

                input_rgb_var = torch.autograd.Variable(input_rgb, requires_grad=False)
                input_flow_var = torch.autograd.Variable(input_flow, requires_grad=False)
                input_pose_var = torch.autograd.Variable(input_pose, requires_grad=False)

                target_var = torch.autograd.Variable(target, requires_grad=False)

                output = model(input_rgb_var, input_flow_var, input_pose_var)
                loss = criterion(output, target_var)

                prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))
                #prec1_rgb, _ = self.accuracy(out_rgb.data, target, topk=(1, 5))
                #prec1_flow, _ = self.accuracy(out_flow.data, target, topk=(1, 5))
                #prec1_pose, _ = self.accuracy(out_pose.data, target, topk=(1, 5))

                losses.update(loss.data, input_rgb.size(0))
                top1.update(prec1.item(), input_rgb.size(0))
                top5.update(prec5.item(), input_rgb.size(0))

                batch_time.update(time.time() - end)
                infer_t += time.time() - start_t
                end = time.time()

                if step % args.logging.print_step == 0:
                    #print("RGB {:.2f} flow {:.2f} Residual {:.2f}".format(prec1_rgb.item(),prec1_flow.item(),prec1_ru.item()))
                    print(('Test: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        step, len(test_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5)))

        self.logger.info(
            'Testing Results: Prec@1 {:.2f} %, Prec@5  {:.2f} %, Loss  {:.4f} '.format(top1.avg, top5.avg, losses.avg))
        # if writer:
        #    n_imgs = min(images.size(0), 10)
        #    for j in range(n_imgs):
        #        writer.add_image('valid/input_image',
        #                         concat_image_features(images[j], first[j]), global_step=step)
        return top1.avg, top5.avg, losses.avg, infer_t

    def adjust_learning_rate(self, optimizer):

        lr = self.optimizer.param_groups[0]['lr']
        wd = self.optimizer.param_groups[0]['weight_decay']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = wd * param_group['decay_mult']
        return self.optimizer.param_groups[0]['lr']


    def save_checkpoint(self, path, epoch, best_prect1, is_best, step_valid_loss):
        state = {"epoch": epoch + 1,
                 "arch": self.args.network.arch,
                 "state_dict": self.model.state_dict(),
                 "optimizer_state": self.optimizer.state_dict(),
                 "scheduler_state": self.scheduler.state_dict(),
                 "best_prec1": best_prect1,
                 'lr': self.optimizer.param_groups[-1]['lr'],
                 'step_valid_loss': step_valid_loss,
                 }
        #save_path = os.path.join(path, 'model_epoch{:03d}.pth'.format(epoch + 1))
        #self.logger.info('Saving model to %s' % save_path)
        #torch.save(state, save_path)
        if is_best:
            best_name = os.path.join(path, 'model_best.pth')
            #shutil.copyfile(save_path, best_name)
            self.logger.info('Saving model_epoch{:03d}.pth to {}'.format(epoch+1, best_name))
            torch.save(state, best_name)


    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def load_finetune(self, path):

        model_dict = self.model.state_dict()
        if not os.path.exists(path):
            print('Path or model file does not exist, can not load pretrained')
            new_state_dict = {}

        else:
            self.logger.info('Loaded model from: ' + path)
            if path is not None:
                pretrained_dict = torch.load(path)
                #print("=> loaded checkpoint '{}' (epoch {})"
                #  .format(path, pretrained_dict['epoch']))
            else:
                print('No default model set')

            new_state_dict = {}
            for k1, v in pretrained_dict['state_dict'].items():
                #print(k1)
                for k2 in model_dict.keys():

                    #k1 = k1.replace('module.base.', 'base.')
                    if k2 in k1 and (v.size() == model_dict[k2].size()):
                        #print("line3")
                        new_state_dict[k2] = v
            print("*" * 50)
            #self.start_epoch = pretrained_dict['epoch']
            un_init_dict_keys = [k for k in model_dict.keys() if k
                                 not in new_state_dict]

            print("un_init_dict_keys: ", un_init_dict_keys)
            print("\n------------------------------------")

            for k in un_init_dict_keys:
                new_state_dict[k] = torch.DoubleTensor(
                    model_dict[k].size()).zero_()
                if 'weight' in k:
                    if 'bn' in k:
                        constant_(new_state_dict[k], 1)
                    else:
                        try:
                            xavier_uniform_(new_state_dict[k])
                        except Exception:
                            constant_(new_state_dict[k], 1)
                elif 'bias' in k:
                    constant_(new_state_dict[k], 0)

            self.model.load_state_dict(new_state_dict)

    def load(self, path):
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch']
        self.best_prec1 = checkpoint['best_prec1']
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))

        if self.args.mode == 'train':
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.step_valid_loss = checkpoint['step_valid_loss']
        #    self.logger.info('Start iter: %d ' % self.start_iter)



    #def test(self, val_loader):
    #    val_loader_iterator = iter(val_loader)
    #    num_val_iters = len(val_loader)
    #    tt = tqdm(range(num_val_iters), total=num_val_iters, desc="Validating")

    #    aux_correct = 0
    #    class_correct = 0
    #    total = 0

    #    self.model.eval()
    #    with torch.no_grad():
    #        for cur_it in tt:
    #            data = next(val_loader_iterator)
    #            if isinstance(data, list):
    #                data = data[0]
    #            # Get the inputs
    #            data = to_device(data, self.device)
    #            imgs = data['images']
    #            cls_lbls = data['class_labels']
    #            aux_lbls = data['aux_labels']

    #            aux_logits, class_logits = self.model(imgs)

    #            _, cls_pred = class_logits.max(dim=1)
    #            _, aux_pred = aux_logits.max(dim=1)

    #            class_correct += torch.sum(cls_pred == cls_lbls.data)
    #            aux_correct += torch.sum(aux_pred == aux_lbls.data)
    #            total += imgs.size(0)

    #        tt.close()

    #    aux_acc = 100 * float(aux_correct) / total
    #    class_acc = 100 * float(class_correct) / total
    #    self.logger.info('aux acc: {:.2f} %, class_acc: {:.2f} %'.format(aux_acc, class_acc))
    #    return aux_acc, class_acc
