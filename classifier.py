import numpy
import os
import shutil
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from stats import AverageMeter

from datetime import datetime

import tblog

def _is_tensor_image(img):
    return torch.is_tensor(img) and (img.ndimension() == 3 or img.ndimension() == 4)

class Classifier():
    def fit(self, labeled_data, unlabeled_data=None, val_data=None):
        raise NotImplementedError

    def predict(self, test_data, accuracy=True):
        raise NotImplementedError

class DLClassifier(Classifier):
    def __init__(self, model, criterion, optimizer='Adam', params=None):
        if not isinstance(model, nn.Module):
             raise RuntimeError('A model must be provided')

        if not isinstance(criterion, nn.Module):
             raise RuntimeError('A criterion must be provided')

        if not optimizer in ['Adam', 'SGD']:
            raise ValueError('Unknown optimizer: {}'.format(optimizer))

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.params = dict()
        self.params['num_workers'] = 4
        self.params['epochs'] = 100
        self.params['start_epoch'] = 0
        self.params['batch_size'] = 32
        self.params['lr'] = 0.001
        self.params['momentum'] = 0.9
        self.params['weight_decay'] = 5e-4
        self.params['lr_steps'] = [50, 90]
        self.params['gamma'] = 0.1
        self.params['print_freq'] = 20
        self.params['resume'] = ''
        self.params['half'] = False
        self.params['save_every'] = 10
        self.params['optimizer'] = model.parameters()

        if params is not None:
            self.params.update(params)

    def fit(self, labeled_data, unlabeled_data=None, val_data=None):

        print(self.model)
        model = torch.nn.DataParallel(self.model).cuda()

        # define loss function (criterion) and optimizer
        criterion = self.criterion.cuda()

        if self.params['half']:
            model.half()
            criterion.half()

        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.params['optimizer'], self.params['lr'],
                                         weight_decay=self.params['weight_decay'])
        else:
            optimizer = torch.optim.SGD(self.params['optimizer'], self.params['lr'],
                                        momentum=self.params['momentum'],
                                        weight_decay=self.params['weight_decay'])

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            gamma=self.params['gamma'],
                                                            milestones=self.params['lr_steps'],
                                                            last_epoch=self.params['start_epoch'] - 1)

        # optionally resume from a checkpoint
        if os.path.isfile(self.params['resume']):
            print("=> loading checkpoint '{}'".format(self.params['resume']))
            checkpoint = torch.load(self.params['resume'])
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            best_prec1 = 0
            start_epoch = self.params['start_epoch']
            print("=> no checkpoint found at '{}'".format(self.params['resume']))


        labeled_loader = DataLoader(labeled_data, drop_last=unlabeled_data is not None,
                             batch_size=self.params['batch_size'], shuffle=True,
                             num_workers=self.params['num_workers'], pin_memory=True)

        unlabeled_loader = None if unlabeled_data is None else DataLoader(unlabeled_data,
                               drop_last=True, batch_size=self.params['batch_size'], shuffle=True,
                               num_workers=self.params['num_workers'], pin_memory=True)

        val_loader = None if val_data is None else DataLoader(val_data,
                         batch_size=self.params['batch_size'], shuffle=False,
                         num_workers=self.params['num_workers'], pin_memory=True)

        writer = tblog.SummaryWriter(tblog.logdir) if tblog.is_enabled else None
        for epoch in range(start_epoch, self.params['epochs']):
            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train_loss, train_prec1 = self._train(labeled_loader, unlabeled_loader, model, criterion, optimizer, epoch)
            lr_scheduler.step()

            # evaluate on validation set
            if val_loader is not None:
                with torch.no_grad():
                    val_loss, val_prec1 = self._validate(val_loader, model, criterion)

                # remember best prec@1 and save checkpoint
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)

            if is_best or epoch % self.params['save_every'] == 0:
                self._save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : lr_scheduler.state_dict(),
                }, is_best, self.params['resume'])

            if tblog.is_enabled:
                writer.add_scalar('Classifier/Loss/train', train_loss, epoch)
                writer.add_scalar('Classifier/Accuracy/train', train_prec1, epoch)
                if val_loader is not None:
                    writer.add_scalar('Classifier/Loss/val', val_loss, epoch)
                    writer.add_scalar('Classifier/Accuracy/val', val_prec1, epoch)

        if tblog.is_enabled:
            writer.close()

        return model.module.state_dict()

    def _train(self, labeled_loader, unlabeled_loader, model, criterion, optimizer, epoch):
        """
            Run one train epoch
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()
        criterion.train()

        if unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)

        end = time.time()
        for i, (labeled_input, target) in enumerate(labeled_loader):

            if unlabeled_loader is not None:
                try:
                    unlabeled_input, _ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_input, _ = next(unlabeled_iter)
                if not _is_tensor_image(unlabeled_input) and _is_tensor_image(labeled_input):
                    unlabeled_input = unlabeled_input.transpose(0, 1).contiguous()
                    unlabeled_input = unlabeled_input.view((-1,) + labeled_input.size()[-3:])
                num_sections = 1 + unlabeled_input.size(0) // labeled_input.size(0)
                input = torch.cat((labeled_input, unlabeled_input), dim=0)
            else:
                input = labeled_input

            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)
            input_var = input # torch.autograd.Variable(input).cuda()
            target_var = target # torch.autograd.Variable(target)
            if self.params['half']:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            if unlabeled_loader is not None:
                labeled_output = torch.split(output, split_size_or_sections=output.size(0) // num_sections, dim=0)[0]
            else:
                labeled_output = output
            prec1 = self._accuracy(labeled_output.data, target)[0]
            losses.update(loss.item(), input.size(0)) # losses.update(loss.data[0], input.size(0))
            top1.update(prec1.item(), input.size(0)) # top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.params['print_freq'] == 0:
                print('[{0}]: '.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')), end='')
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(labeled_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1))

        return losses.avg, top1.avg

    def _validate(self, loader, model, criterion):
        """
        Run evaluation
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        criterion.eval()

        end = time.time()
        for i, (input, target) in enumerate(loader):
            target = target.cuda(non_blocking=True)
            input_var = input # torch.autograd.Variable(input, volatile=True).cuda()
            target_var = target # torch.autograd.Variable(target, volatile=True)

            if self.params['half']:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = self._accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0)) # losses.update(loss.data[0], input.size(0))
            top1.update(prec1.item(), input.size(0)) # top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.params['print_freq'] == 0:
                print('[{0}]: '.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')), end='')
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))

        return losses.avg, top1.avg

    def _save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """
        Save the training model
        """
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, filename.replace('checkpoint', 'model_best'))

    def _accuracy(self, output, target, topk=(1,)):
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

    def predict(self, test_data, accuracy=True):

        print(self.model)
        model = torch.nn.DataParallel(self.model).cuda()

        # define loss function (criterion) and optimizer
        criterion = self.criterion.cuda()

        if self.params['half']:
            model.half()
            criterion.half()

        test_loader = DataLoader(test_data,
                                 batch_size=self.params['batch_size'], shuffle=False,
                                 num_workers=self.params['num_workers'], pin_memory=True)

        with torch.no_grad():
            if accuracy:
                _, predictions = self._validate(test_loader, model, criterion)
            else:
                predictions = self._test(test_loader, model)

        return predictions

    def _test(self, loader, model):
        """
        Run evaluation
        """
        batch_time = AverageMeter()

        # switch to evaluate mode
        model.eval()

        predictions = []
        end = time.time()
        for i, (input, target) in enumerate(loader):
            input_var = input # torch.autograd.Variable(input, volatile=True).cuda()

            if self.params['half']:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            output = torch.softmax(output, dim=1)

            output = output.float()
            predictions.extend(output.cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.params['print_freq'] == 0:
                print('[{0}]: '.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')), end='')
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time))

        return numpy.stack(predictions)
