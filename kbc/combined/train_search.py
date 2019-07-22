import os
import sys
import time
import glob
import tqdm
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch import optim
from typing import Dict
from datasets import Dataset
from regularizers import N2, N3, Regularizer
from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--reduction', action='store_true', help='use reduction cells in convnet')
parser.add_argument('--steps', type=int, default=4, help='number of steps in learned cell')

datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)
# models = ['CP', 'ComplEx']
# parser.add_argument(
#     '--model', choices=models,
#     help="Model in {}".format(models)
# )
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
optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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

  dataset = Dataset(args.dataset)
  train_examples = torch.from_numpy(dataset.get_train().astype('int64'))
  valid_examples = torch.from_numpy(dataset.get_valid().astype('int64'))

  CLASSES = dataset.get_shape()[0]

  criterion = nn.CrossEntropyLoss(reduction='mean')
  criterion = criterion.cuda()

  regularizer = {
      'N2': N2(args.reg),
      'N3': N3(args.reg),
    }[args.regularizer]

  #TODO there are some default kwargs in network we're not currently setting
  model = Network(args.init_channels, CLASSES, args.layers, criterion, 
    regularizer, dataset.get_shape(), args.rank, args.init, args.reduction, args.steps)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  #TODO force SGD for now, can we change?
  optimizer = {
    'Adagrad': lambda: optim.Adagrad(
      model.parameters(), 
      lr=args.learning_rate,
      #momentum=args.momentum,
      weight_decay=args.weight_decay),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
  }[args.optimizer]()

  # optimizer = torch.optim.SGD(
  #     model.parameters(),
  #     args.learning_rate,
  # #TODO can we reintroduce these?
  #     momentum=args.momentum,
  #     weight_decay=args.weight_decay)

  #train_transform, valid_transform = utils._data_transforms_cifar10(args)
  #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  #num_train = len(train_data)
  #indices = list(range(num_train))
  #split = int(np.floor(args.train_portion * num_train))

  #TODO use queues?

  train_queue = torch.utils.data.DataLoader(
      train_examples, batch_size=args.batch_size,
      shuffle = True,
      #sampler=torch.utils.data.sampler.RandomSampler(),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_examples, batch_size=args.batch_size,
      shuffle = True,
      #sampler=torch.utils.data.sampler.RandomSampler(),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  best_acc = 0
  curve = {'train': [], 'valid': [], 'test': []}

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    if args.reduction:
      print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    #was using train_queue/valid_queue here instead of examples
    #removed acc/obj here as in original code
    train_epoch(train_examples, train_queue, valid_queue, model, 
      architect, criterion, optimizer, regularizer, args.batch_size, args.learning_rate)
    #logging.info('train_acc %f', train_acc)

    # validation
    # valid_acc, valid_obj = infer(valid_queue, model, criterion)
    # logging.info('valid_acc %f', valid_acc)
    if (epoch + 1) % args.report_freq == 0:
      valid, test, train = [
              avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
              for split in ['valid', 'test', 'train']
          ]
      curve['valid'].append(valid)
      curve['test'].append(test)
      curve['train'].append(train)

      print("\t TRAIN: ", train)
      print("\t VALID : ", valid)

      is_best = False
      if valid['MRR'] > best_acc:
        best_acc = valid['MRR']
        is_best = True

    utils.save(model, os.path.join(args.save, 'weights.pt'))


# def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
#   objs = utils.AvgrageMeter()
#   top1 = utils.AvgrageMeter()
#   top5 = utils.AvgrageMeter()

#   for step, (input, target) in enumerate(train_queue):
#     model.train()
#     n = input.size(0)

#     input = Variable(input, requires_grad=False).cuda()
#     target = Variable(target, requires_grad=False).cuda(async=True)

#     # get a random minibatch from the search queue with replacement
#     input_search, target_search = next(iter(valid_queue))
#     input_search = Variable(input_search, requires_grad=False).cuda()
#     target_search = Variable(target_search, requires_grad=False).cuda(async=True)

#     architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

#     optimizer.zero_grad()
#     logits = model(input)
#     loss = criterion(logits, target)

#     loss.backward()
#     nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
#     optimizer.step()

#     prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#     objs.update(loss.data[0], n)
#     top1.update(prec1.data[0], n)
#     top5.update(prec5.data[0], n)

#     if step % args.report_freq == 0:
#       logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

# return top1.avg, objs.avg

def train_epoch(train_examples,train_queue, valid_queue,
  model, architect, criterion, optimizer: optim.Optimizer, 
  regularizer: Regularizer, batch_size: int, lr, verbose: bool = True):
  #train_examples = train_examples[torch.randperm(train_examples.shape[0]), :]
  #valid_examples = valid_examples[torch.randperm(valid_examples.shape[0]), :]
  loss = nn.CrossEntropyLoss(reduction='mean')
  with tqdm.tqdm(total=train_examples.shape[0], unit='ex', disable=not verbose) as bar:
      bar.set_description(f'train loss')
      #b_begin = 0
      #while b_begin < valid_examples.shape[0]:
      for step, input in enumerate(train_queue):
          ##set current batch
          # input = train_examples[
          #     b_begin:b_begin + batch_size
          # ].cuda()

          # input_search = valid_examples[
          # b_begin:b_begin + batch_size
          # ].cuda()

          # target = input[:, 2]
          # target_search = input_search[:, 2]

          model.train()
          n = input.size(0)

          #TODO is this okay???
          input = Variable(input, requires_grad=False).cuda()
          target = Variable(input[:,2], requires_grad=False).cuda()#async=True)

          # get a random minibatch from the search queue with replacement
          input_search = next(iter(valid_queue))
          input_search = Variable(input_search, requires_grad=False).cuda()
          target_search = Variable(input_search[:,2], requires_grad=False).cuda()#async=True)

          architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

          #compute predictions, ground truth
          predictions, factors = model.forward(input)
          truth = input[:, 2]

          #evaluate loss
          l_fit = loss(predictions, truth)
          l_reg = regularizer.forward(factors)
          l = l_fit + l_reg

          #optimise
          optimizer.zero_grad()
          l.backward()
          nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
          optimizer.step()
          #b_begin += batch_size

          #progress bar
          bar.update(input.shape[0])
          bar.set_postfix(loss=f'{l.item():.0f}')

          #TODO from DARTS train fn - grad clipping, accuracy metrics?
          
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


# def infer(valid_queue, model, criterion):
#   objs = utils.AvgrageMeter()
#   top1 = utils.AvgrageMeter()
#   top5 = utils.AvgrageMeter()
#   model.eval()

#   for step, (input, target) in enumerate(valid_queue):
#     input = Variable(input, volatile=True).cuda()
#     target = Variable(target, volatile=True).cuda(async=True)

#     logits = model(input)
#     loss = criterion(logits, target)

#     prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#     n = input.size(0)
#     objs.update(loss.data[0], n)
#     top1.update(prec1.data[0], n)
#     top5.update(prec5.data[0], n)

#     if step % args.report_freq == 0:
#       logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

#   return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

