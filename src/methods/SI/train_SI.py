import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import copy
import os
import pdb
import shutil
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch import Tensor
from typing import List
import utilities.utils as utils


class Elastic_SGD(optim.SGD):
    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(Elastic_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(Elastic_SGD, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # print('************************DOING A STEP************************')
        # loss=super(Elastic_SGD, self).step(closure)
        loss = None
        if closure is not None:
            loss = closure()
        index = 0
        reg_lambda = reg_params.get('lambda')
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                # print('************************ONE PARAM************************')

                if p.grad is None:
                    continue

                d_p = p.grad.data
                unreg_dp = p.grad.data.clone()
                # HERE MY CODE GOES
                reg_param = reg_params.get(p)

                omega = reg_param.get('omega')
                zero = torch.FloatTensor(p.data.size()).zero_()
                init_val = reg_param.get('init_val')
                w = reg_param.get('w')
                curr_wegiht_val = p.data.clone()
                # move the variables to cuda
                init_val = init_val.cuda()
                w = w.cuda()
                omega = omega.cuda()
                # get the difference
                weight_dif = curr_wegiht_val.add(-1, init_val)

                regulizer = torch.mul(weight_dif, 2 * reg_lambda * omega)

                # JUST NOW PUT BACK
                d_p.add_(regulizer)
                del weight_dif
                del omega

                del regulizer
                # HERE MY CODE ENDS

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    # pdb.set_trace()
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                #
                p.data.add_(-group['lr'], d_p)
                w_diff = p.data.add(-1, curr_wegiht_val)

                del curr_wegiht_val

                change = w_diff.mul(unreg_dp)
                del unreg_dp
                change = torch.mul(change, -1)
                del w_diff
                if 0:
                    if change.equal(zero.cuda()):
                        print('change zero')
                        pdb.set_trace()
                    if w.equal(zero.cuda()):
                        print('w zero')

                    if w.equal(zero.cuda()):
                        print('w after zero')
                    x = p.data.add(-init_val)
                    if x.equal(zero.cuda()):
                        print('path diff is zero')
                del zero
                del init_val
                w.add_(change)
                reg_param['w'] = w
                # after deadline

                reg_params[p] = reg_param
                index += 1
        return loss


class Elastic_RmsPropOptimizer(optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(Elastic_RmsPropOptimizer, self).__init__(params, lr, alpha, eps, weight_decay, momentum, centered)

    def __setstate__(self, state):
        super(Elastic_RmsPropOptimizer, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.

                Arguments:
                    closure (callable, optional): A closure that reevaluates the model
                        and returns the loss.
                """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        index = 0
        reg_lambda = reg_params.get('lambda')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # HERE MY CODE GOES, HYX
                d_p = p.grad.data
                unreg_dp = p.grad.data.clone()

                reg_param = reg_params.get(p)

                omega = reg_param.get('omega')
                # zero = torch.FloatTensor(p.data.size()).zero_()
                init_val = reg_param.get('init_val')
                w = reg_param.get('w')
                curr_wegiht_val = p.data.clone()
                # move the variables to cuda
                init_val = init_val.cuda()
                w = w.cuda()
                omega = omega.cuda()
                # get the difference
                weight_dif = curr_wegiht_val.add(-1, init_val)

                regulizer = torch.mul(weight_dif, 2 * reg_lambda * omega)
                # print("HYX, regularizer sum: index ", index,
                #       "regularizer", torch.sum(regulizer.clone()).item(),
                #       "curr_weight", torch.sum(curr_wegiht_val).item(),
                #       "weight_dif", torch.sum(weight_dif.clone()).item(),
                #       "omega", torch.sum(omega.clone()).item(),
                #       "lambda", reg_lambda)
                # JUST NOW PUT BACK

                # print("HYX, gradient sum before operation:", torch.sum(p.grad.data.clone()), index)
                d_p.add_(regulizer)
                # p.grad.data.add_(regulizer)
                # print("HYX, gradient sum after operation:", torch.sum(p.grad.data.clone()), index)
                del weight_dif
                del omega

                del regulizer
                # HERE MY CODE ENDS

                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    # moment_buff = torch.sum(buf).item()
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    # moment_buff2 = torch.sum(buf).item()
                    # p.add_(buf, alpha=-group['lr']) # original
                    # before = torch.sum(p.data).item()
                    p.data.add_(buf, alpha=-group['lr']) #HYX
                    # after = torch.sum(p.data).item()
                    # print("HYX, indx ", index,
                    #       "momentum_buffer", moment_buff, "buff2", moment_buff2,
                    #       "p before", before, "p after", after, "learning rate", -group['lr'], "avg", torch.sum(avg).item())
                else:
                    p.data.addcdiv_(grad, avg, value=-group['lr'])

                # update w  HYX
                w_diff = p.data.clone().add(-1, curr_wegiht_val)

                del curr_wegiht_val

                change = w_diff.mul(unreg_dp)
                del unreg_dp
                change = torch.mul(change, -1)
                del w_diff
                # del zero
                del init_val
                w.add_(change)
                reg_param['w'] = w
                # after deadline

                reg_params[p] = reg_param
                index += 1
        return loss


class Elastic_Adam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(Elastic_Adam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def __setstate__(self, state):
        super(Elastic_Adam, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # index = 0
        reg_lambda = reg_params.get('lambda')
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')


                    # HERE MY CODE GOES, HYX
                    d_p = p.grad.data
                    reg_param = reg_params.get(p)
                    omega = reg_param.get('omega')
                    init_val = reg_param.get('init_val')
                    curr_wegiht_val = p.data.clone()
                    init_val = init_val.cuda()
                    omega = omega.cuda()
                    weight_dif = curr_wegiht_val.add(-1, init_val)
                    regulizer = torch.mul(weight_dif, 2 * reg_lambda * omega)
                    #
                    # print("HYX, gradient sum before operation:", torch.sum(p.grad.data.clone()).item(), index)
                    d_p.add_(regulizer)
                    # p.grad.data.add_(regulizer)
                    # print("HYX, gradient sum after operation:", torch.sum(p.grad.data.clone()).item(), index)
                    del weight_dif, omega, regulizer, curr_wegiht_val, init_val
                    # HERE MY CODE ENDS
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            adamSI(reg_params,
                   params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )
        return loss

def adamSI(reg_params,
         params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):
        unreg_dp = param.grad.data.clone()
        reg_param = reg_params.get(param)
        w = reg_param.get('w')
        w = w.cuda()
        curr_wegiht_val = param.data.clone()


        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.data.addcdiv_(exp_avg, denom, value=-step_size)

        w_diff = param.data.add(-1, curr_wegiht_val)
        del curr_wegiht_val

        change = w_diff.mul(unreg_dp)
        del unreg_dp
        change = torch.mul(change, -1)
        del w_diff
        w.add_(change)
        del change
        reg_param['w'] = w
        reg_params[param] = reg_param


class Original_Adam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(Original_Adam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def __setstate__(self, state):
        super(Original_Adam, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            adamOriginal(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )
        return loss

def adamOriginal(
         params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1
        param.data.addcdiv_(exp_avg, denom, value=-step_size)


def set_lr(optimizer, lr, count):
    """Decay learning rate by a factor of 0.5 every lr_decay_epoch epochs."""
    continue_training = True
    if count >= 20:
        continue_training = False
        print("training terminated")
    if count == 11:
        lr = lr * 0.5
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training


def traminate_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    utils.print_timing(time_elapsed, "TRAINING ONLY")
    print('Best val Acc: {:4f}'.format(best_acc))


def train_model(model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu, num_epochs, exp_dir='./',
                resume='', previous_model_path='', saving_freq=5, reload_optimizer=True):
    print('dictoinary length' + str(len(dset_loaders)))
    since = time.time()
    val_beat_counts = 0  # number of time val accuracy not imporved
    best_model = model
    best_acc = 0.0
    mem_snapshotted = False

    if os.path.isfile(resume):

        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']
        print("val_beat_counts", val_beat_counts)
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    elif os.path.isfile(previous_model_path) and reload_optimizer:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))
        print("load checkpoint from previous task/center model at '{}'".format(previous_model_path))
        checkpoint = torch.load(previous_model_path)
        # model.load_state_dict(checkpoint['state_dict']) # has already been loaded
        lr = checkpoint['lr']
        print("lr is ", lr)
        print('load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(previous_model_path, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    warning_NAN_counter = 0
    print("START EPOCH = ", str(start_epoch))
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    traminate_protocol(since, best_acc)
                    return model, best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # print('step')
                    optimizer.step(model.reg_params)

                # statistics
                if math.isnan(loss.data.item()):
                    warning_NAN_counter += 1

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]

            if warning_NAN_counter > 0:
                print("SKIPPED NAN RUNNING LOSS FOR BATCH: ", warning_NAN_counter, " TIMES")
            print("EPOCH LOSS=", epoch_loss, ", RUNNING LOSS=", running_loss, ",DIVISION=", dset_sizes[phase])
            if epoch_loss > 1e4 or math.isnan(epoch_loss):
                print("TERMINATING: Epoch loss [", epoch_loss, "]  is NaN or > 1e4")
                return model, best_acc
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del outputs
                    del labels
                    del inputs
                    del loss
                    del preds
                    best_acc = epoch_acc
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    print('new best val model')
                    epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
                    save_checkpoint({
                        'epoch_acc': epoch_acc,
                        'best_acc': best_acc,
                        'epoch': epoch,
                        'lr': lr,
                        'val_beat_counts': val_beat_counts,
                        'arch': 'alexnet',
                        'model': model,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch_file_name)

                    val_beat_counts = 0
                else:
                    val_beat_counts += 1

                if epoch == num_epochs:
                    epoch_file_name = exp_dir + '/' + 'last_epoch' + '.pth.tar'
                    save_checkpoint({
                        'epoch_acc': epoch_acc,
                        'best_acc': best_acc,
                        'epoch': epoch,
                        'lr': lr,
                        'val_beat_counts': val_beat_counts,
                        'arch': 'alexnet',
                        'model': model,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch_file_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc  # initialize importance dictionary


def initialize_reg_params(model):
    reg_params = {}
    for name, param in model.named_parameters():  # after deadline check
        w = torch.FloatTensor(param.size()).zero_()
        omega = torch.FloatTensor(param.size()).zero_()
        init_val = param.data.clone()
        reg_param = {}
        reg_param['omega'] = omega
        reg_param['w'] = w
        reg_param['init_val'] = init_val
        reg_param['name'] = name
        reg_params[param] = reg_param
    return reg_params


def update_reg_params(model, slak=1e-3):
    reg_params = model.reg_params
    index = 0
    for param in list(model.parameters()):
        print('index' + str(index))
        if param in reg_params.keys():
            print('updating index' + str(index))
            reg_param = reg_params.get(param)
            w = reg_param.get('w').cuda()
            zero = torch.FloatTensor(param.data.size()).zero_()

            if w.equal(zero.cuda()):
                print('W IS WRONG WARNING')
            omega = reg_param.get('omega')
            omega = omega.cuda()
            if not omega.equal(zero.cuda()):
                print('omega is not equal zero')
            else:
                print('omega is equal zero')

            omega = omega.cuda()
            init_val = reg_param.get('init_val')
            init_val = init_val.cuda()
            path_diff = param.data.add(-1, init_val)
            if path_diff.equal(zero.cuda()):
                print('PATH DIFF WRONG WARNING')
            dominator = path_diff.pow(2)
            dominator.add_(slak)
            this_omega = w.div(dominator)

            ####
            if 0:
                the_size = 1
                for x in this_omega.size():
                    the_size = the_size * x
                om = this_omega.view(the_size)
                randindex = torch.randperm(the_size)
                om = om[randindex.cuda()]
                this_omega = om.view(this_omega.size())

            this_omega = torch.max(this_omega, zero.cuda())
            print("**********max*************")
            print(this_omega.max())
            print("**********min*************")
            print(this_omega.min())
            omega.add_(this_omega)

            reg_param['omega'] = omega
            w = zero.cuda()
            reg_param['w'] = w
            reg_param['init_val'] = param.data.clone()
            reg_params[param] = reg_param
        else:
            print('initializing index' + str(index))
            w = torch.FloatTensor(param.size()).zero_()
            omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param = {}
            reg_param['omega'] = omega
            reg_param['w'] = w
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
        index = index + 1
    return reg_params


def update_reg_params(model, slak=1e-3):
    reg_params = model.reg_params
    index = 0
    for param in list(model.parameters()):
        print('index' + str(index))
        if param in reg_params.keys():
            print('updating index' + str(index))
            reg_param = reg_params.get(param)
            w = reg_param.get('w').cuda()
            zero = torch.FloatTensor(param.data.size()).zero_()

            if w.equal(zero.cuda()):
                print('W IS WRONG WARNING')
            omega = reg_param.get('omega')
            omega = omega.cuda()
            if not omega.equal(zero.cuda()):
                print('omega is not equal zero')
            else:
                print('omega is equal zero')

            omega = omega.cuda()
            init_val = reg_param.get('init_val')
            init_val = init_val.cuda()
            path_diff = param.data.add(-1, init_val)
            if path_diff.equal(zero.cuda()):
                print('PATH DIFF WRONG WARNING')
            dominator = path_diff.pow(2)
            dominator.add_(slak)
            this_omega = w.div(dominator)

            ####
            if 0:
                the_size = 1
                for x in this_omega.size():
                    the_size = the_size * x
                om = this_omega.view(the_size)
                randindex = torch.randperm(the_size)
                om = om[randindex.cuda()]
                this_omega = om.view(this_omega.size())

            this_omega = torch.max(this_omega, zero.cuda())
            print("**********max*************")
            print(this_omega.max())
            print("**********min*************")
            print(this_omega.min())
            omega.add_(this_omega)

            reg_param['omega'] = omega
            w = zero.cuda()
            reg_param['w'] = w
            reg_param['init_val'] = param.data.clone()
            reg_params[param] = reg_param
        else:
            print('initializing index' + str(index))
            w = torch.FloatTensor(param.size()).zero_()
            omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param = {}
            reg_param['omega'] = omega
            reg_param['w'] = w
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
        index = index + 1
    return reg_params


def update_reg_params_ref(model, slak=1e-3):
    reg_params = model.reg_params
    new_reg_params = {}
    index = 0
    for param in list(model.parameters()):
        print('index' + str(index))

        reg_param = reg_params.get(param)
        w = reg_param.get('w')
        zero = torch.FloatTensor(param.data.size()).zero_()
        if w.equal(zero.cuda()):
            print('wrong')
        omega = reg_param.get('omega')
        omega = omega.cuda()
        if not omega.equal(zero.cuda()):
            print('omega wrong')

        omega = omega.cuda()
        init_val = reg_param.get('init_val')
        init_val = init_val.cuda()
        path_diff = param.data.add(-init_val)
        if path_diff.equal(zero.cuda()):
            print('path_diff wrong')
        dominator = path_diff.pow_(2)
        dominator.add_(slak)
        this_omega = w.div(dominator)
        if this_omega.equal(zero.cuda()):
            print('this_omega wrong')
        omega.add_(this_omega)
        reg_param['omega'] = omega
        reg_param['w'] = w
        reg_param['init_val'] = param.data
        reg_params[param] = reg_param
        model.reg_params = reg_params
        new_reg_params[index] = reg_param
        index = index + 1
    return new_reg_params


def reassign_reg_params(model, new_reg_params):
    reg_params = model.reg_params
    reg_params = {}
    index = 0
    for param in list(model.parameters()):
        reg_param = new_reg_params[index]
        reg_params[param] = reg_param
        index = index + 1
    model.reg_params = reg_params
    return reg_params


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
