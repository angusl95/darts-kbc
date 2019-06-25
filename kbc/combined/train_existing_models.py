import os
import sys
import numpy as np
import time
import tqdm
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch import optim
from typing import Dict
from datasets import Dataset
from models import CP, ComplEx
from regularizers import N2, N3, Regularizer

from torch.autograd import Variable
from model import NetworkImageNet as Network


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')

datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)
models = ['CP', 'ComplEx']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
regularizers = ['N3', 'N2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)


args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

print('dataset shape')
print(dataset.get_shape())
model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
}[args.model]()

regularizer = {
    'N2': N2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

device = 'cuda'
model.to(device)

#check this
CLASSES = dataset.get_shape()[0]

print('num classes:', CLASSES)

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # genotype = eval("genotypes.%s" % args.arch)
  # #model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  # if args.parallel:
  #   model = nn.DataParallel(model).cuda()
  # else:
  #   model = model.cuda()

  # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  # traindir = os.path.join(args.data, 'train')
  # validdir = os.path.join(args.data, 'val')
  # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  # train_data = dset.ImageFolder(
  #   traindir,
  #   transforms.Compose([
  #     transforms.RandomResizedCrop(224),
  #     transforms.RandomHorizontalFlip(),
  #     transforms.ColorJitter(
  #       brightness=0.4,
  #       contrast=0.4,
  #       saturation=0.4,
  #       hue=0.2),
  #     transforms.ToTensor(),
  #     normalize,
  #   ]))
  # valid_data = dset.ImageFolder(
  #   validdir,
  #   transforms.Compose([
  #     transforms.Resize(256),
  #     transforms.CenterCrop(224),
  #     transforms.ToTensor(),
  #     normalize,
  #   ]))

  # train_queue = torch.utils.data.DataLoader(
  #   train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

  # valid_queue = torch.utils.data.DataLoader(
  #   valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

  best_acc_top1 = 0
  # for epoch in range(args.epochs):
  #   scheduler.step()
  #   logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
  #   model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

  #   train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
  #   logging.info('train_acc %f', train_acc)

  #   valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
  #   logging.info('valid_acc_top1 %f', valid_acc_top1)
  #   logging.info('valid_acc_top5 %f', valid_acc_top5)

  #   is_best = False
  #   if valid_acc_top1 > best_acc_top1:
  #     best_acc_top1 = valid_acc_top1
  #     is_best = True

  #   utils.save_checkpoint({
  #     'epoch': epoch + 1,
  #     'state_dict': model.state_dict(),
  #     'best_acc_top1': best_acc_top1,
  #     'optimizer' : optimizer.state_dict(),
  #     }, is_best, args.save)

  cur_loss = 0
  curve = {'train': [], 'valid': [], 'test': []}
  for e in range(args.epochs):
    cur_loss = train_epoch(examples, model, optimizer, regularizer, args.batch_size)

    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)
  results = dataset.eval(model, 'test', -1)
  print("\n\nTEST : ", results)

def train_epoch(examples: torch.LongTensor, model, optimizer: optim.Optimizer, 
  regularizer: Regularizer, batch_size: int, verbose: bool = True):
  actual_examples = examples[torch.randperm(examples.shape[0]), :]
  loss = nn.CrossEntropyLoss(reduction='mean')
  with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not verbose) as bar:
      bar.set_description(f'train loss')
      b_begin = 0
      while b_begin < examples.shape[0]:
          ##set current batch
          input_batch = actual_examples[
              b_begin:b_begin + batch_size
          ].cuda()

          #compute predictions, ground truth
          predictions, factors = model.forward(input_batch)
          truth = input_batch[:, 2]

          #evaluate loss
          l_fit = loss(predictions, truth)
          l_reg = regularizer.forward(factors)
          l = l_fit + l_reg

          #optimise
          optimizer.zero_grad()
          l.backward()
          optimizer.step()
          b_begin += batch_size

          #progress bar
          bar.update(input_batch.shape[0])
          bar.set_postfix(loss=f'{l.item():.0f}')  


# def train(train_queue, model, criterion, optimizer):
#   objs = utils.AvgrageMeter()
#   top1 = utils.AvgrageMeter()
#   top5 = utils.AvgrageMeter()
#   model.train()

#   for step, (input, target) in enumerate(train_queue):
#     target = target.cuda(async=True)
#     input = input.cuda()
#     input = Variable(input)
#     target = Variable(target)

#     optimizer.zero_grad()
#     logits, logits_aux = model(input)
#     loss = criterion(logits, target)
#     if args.auxiliary:
#       loss_aux = criterion(logits_aux, target)
#       loss += args.auxiliary_weight*loss_aux

#     loss.backward()
#     nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
#     optimizer.step()

#     prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#     n = input.size(0)
#     objs.update(loss.data[0], n)
#     top1.update(prec1.data[0], n)
#     top5.update(prec5.data[0], n)

#     if step % args.report_freq == 0:
#       logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

#   return top1.avg, objs.avg


# def infer(valid_queue, model, criterion):
#   objs = utils.AvgrageMeter()
#   top1 = utils.AvgrageMeter()
#   top5 = utils.AvgrageMeter()
#   model.eval()

#   for step, (input, target) in enumerate(valid_queue):
#     input = Variable(input, volatile=True).cuda()
#     target = Variable(target, volatile=True).cuda(async=True)

#     logits, _ = model(input)
#     loss = criterion(logits, target)

#     prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#     n = input.size(0)
#     objs.update(loss.data[0], n)
#     top1.update(prec1.data[0], n)
#     top5.update(prec5.data[0], n)

#     if step % args.report_freq == 0:
#       logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

#   return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 
