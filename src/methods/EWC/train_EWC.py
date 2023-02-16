import math
import os
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import Tensor
from typing import List
import utilities.utils as utils


class Weight_Regularized_SGD(optim.SGD):
    r"""Implements stochastic gradient descent with an EWC penalty on important weights for previous tasks
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(Weight_Regularized_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            reg_params: a dictionary where importance weights for each parameter is stored.
        """

        loss = None
        if closure is not None:
            loss = closure()
        index = 0
        reg_lambda = reg_params.get('lambda')  # a hyper parameter for the EWC regularizer

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # This part is to add the gradients of the EWC regularizer

                if p in reg_params:
                    # for each parameter considered in the optimization process
                    reg_param = reg_params.get(
                        p)  # get the corresponding dictionary where the information for EWC penalty is stored
                    omega = reg_param.get('omega')  # the accumelated Fisher information matrix.
                    init_val = reg_param.get('init_val')  # theta*, the optimal parameters up until the previous task.
                    curr_wegiht_val = p.data  # get the current weight value
                    # move the variables to cuda
                    init_val = init_val.cuda()
                    omega = omega.cuda()

                    # get the difference
                    weight_dif = curr_wegiht_val.add(-1, init_val)  # compute the difference between theta and theta*,

                    regulizer = weight_dif.mul(2 * reg_lambda * omega)  # the gradient of the EWC penalty
                    d_p.add_(regulizer)  # add the gradient of the penalty

                    # delete unused variables
                    del weight_dif, curr_wegiht_val, omega, init_val, regulizer
                # The EWC regularizer ends here
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
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

                p.data.add_(-group['lr'], d_p)
                index += 1
        return loss


class Weight_Regularized_Adam(optim.Adam):
    r"""Implements stochastic gradient descent with an EWC penalty on important weights for previous tasks
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(Weight_Regularized_Adam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def __setstate__(self, state):
        super(Weight_Regularized_Adam, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            reg_params: a dictionary where importance weights for each parameter is stored.
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
                    del weight_dif, curr_wegiht_val, omega, init_val, regulizer
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
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training = True
    if count > 20:
        continue_training = False
        print("training terminated")
    if count == 10:
        lr = lr * 0.5
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training


def traminate_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def train_model(model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu, num_epochs, exp_dir='./',
                resume='', previous_model_path='', saving_freq=5, reload_optimizer=True):
    """
    Trains a deep learning model with EWC penalty.
    Empirical Fisher is used instead of true FIM for efficiency and as there are no significant differences in results.
    :return: last model & best validation accuracy
    """

    print('dictoinary length' + str(len(dset_loaders)))
    since = time.time()
    mem_snapshotted = False
    val_beat_counts = 0  # number of time val accuracy not imporved
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']

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

    print(str(start_epoch))
    print("lr is", lr)
    for epoch in range(start_epoch, num_epochs + 2):
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
                # FOR MNIST DATASET
                # inputs = inputs.squeeze()

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # call the optimizer and pass reg_params to be utilized in the EWC penalty
                    optimizer.step(model.reg_params)

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if epoch_loss > 1e4 or math.isnan(epoch_loss):
                return model, best_acc
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del outputs, labels, inputs, loss, preds
                    best_acc = epoch_acc
                    print("new best val accuracy")
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))

                    epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
                    save_checkpoint({
                        'epoch_acc': epoch_acc,
                        'best_acc': best_acc,
                        'epoch': epoch + 1,
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
                        'epoch': epoch + 1,
                        'lr': lr,
                        'val_beat_counts': val_beat_counts,
                        'arch': 'alexnet',
                        'model': model,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
