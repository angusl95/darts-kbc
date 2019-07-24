import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores > targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      #TODO reintroduce this?
      #if 'pool' in primitive:
      #  op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    # if reduction_prev:
    #   self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    # else:
    #   self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    # self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    # s0 = self.preprocess0(s0)
    # s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(KBCModel):

  def __init__(self, C, num_classes, layers, criterion, regularizer, 
    sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3,
    reduction_flag = True, steps=4, multiplier=4, stem_multiplier=3):
    #TODO: remove stem multiplier from args?
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._regularizer = regularizer
    self._steps = steps
    self._multiplier = multiplier
    self._stem_multiplier = stem_multiplier
    self.rank = rank
    self.sizes = sizes
    self._init_size = init_size
    self._reduction_flag = reduction_flag
    self.embeddings = nn.ModuleList([
      #TODO restore sparse here?
            nn.Embedding(s, rank)#, sparse=True)
            for s in sizes[:2]
        ])
    self.embeddings[0].weight.data *= init_size
    self.embeddings[1].weight.data *= init_size

    C_curr = C
    #C_curr = stem_multiplier*C
    # self.stem = nn.Sequential(
    #   nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
    #   nn.BatchNorm2d(C_curr)
    # )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if self._reduction_flag:
        if i in [layers//3, 2*layers//3]:
          C_curr *= 2
          reduction = True
        else:
          reduction = False
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.projection = nn.Linear(C_prev, self.rank, bias=False)
    #self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._regularizer,
      self.sizes, self.rank, self._init_size, self._reduction_flag, self._steps, 
      self._multiplier, self._stem_multiplier).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def score(self, x):
    lhs = self.embeddings[0](x[:, 0])
    rel = self.embeddings[1](x[:, 1])
    rhs = self.embeddings[0](x[:, 2])
    
    to_score = self.embeddings[0].weight
    lhs = lhs.view([lhs.size(0),1,16,self.rank//16])
    rel = rel.view([rel.size(0),1,16,self.rank//16])
    combined = torch.cat([lhs,rel],3)
    input = combined.view([lhs.size(0),1,32,-1])
    #input = torch.cat([lhs, rel], 1).view([lhs.size(0), 3, 16, (self.rank * 2)//(16*3)])
    s0 = s1 = input

    for i, cell in enumerate(self.cells):
      if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
          weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    out = s1
    out = self.projection(out.view(out.size(0),-1))
    out = F.relu(out)
    out = torch.sum(
        out * rhs, 1, keepdim=True
    )
    return out

  def forward(self, x):
    lhs = self.embeddings[0](x[:, 0])
    rel = self.embeddings[1](x[:, 1])
    rhs = self.embeddings[0](x[:, 2])


    to_score = self.embeddings[0].weight
    lhs = lhs.view([lhs.size(0),1,16,self.rank//16])
    rel = rel.view([rel.size(0),1,16,self.rank//16])
    combined = torch.cat([lhs,rel],3)
    input = combined.view([lhs.size(0),1,32,-1])

    #input = torch.cat([lhs, rel], 1).view([lhs.size(0), 3, 16, (self.rank * 2)//(16*3)])
    s0 = s1 = input
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    # logits = self.classifier(out.view(out.size(0),-1))
    # return logits
    out = s1
    out = self.projection(out.view(out.size(0),-1))
    out = F.relu(out)
    out = out @ to_score.transpose(0,1)
    return out, (lhs,rel,rhs)

  def get_rhs(self, chunk_begin: int, chunk_size: int):
    return self.embeddings[0].weight.data[
        chunk_begin:chunk_begin + chunk_size
    ].transpose(0, 1)

  def get_queries(self, queries: torch.Tensor):
    lhs = self.embeddings[0](queries[:, 0])
    rel = self.embeddings[1](queries[:, 1])
    lhs = lhs.view([lhs.size(0),1,16,self.rank//16])
    rel = rel.view([rel.size(0),1,16,self.rank//16])
    combined = torch.cat([lhs,rel],3)
    input = combined.view([lhs.size(0),1,32,-1])
    #input = torch.cat([lhs, rel], 1).view([lhs.size(0), 3, 16, (self.rank * 2)//(16*3)])
    s0 = s1 = input

    for i, cell in enumerate(self.cells):
      if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
          weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    out = s1
    out = self.projection(out.view(out.size(0),-1))
    out = F.relu(out)

    return out

  def _loss(self, input, target):  
    #TODO: definitely just alias for forward method?
    logits, factors = self(input)
    l_fit = self._criterion(logits, target)
    l_reg = self._regularizer.forward(factors)
    #return self._criterion(logits, target) 
    return l_fit + l_reg

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    if self._reduction_flag:
      self._arch_parameters = [
        self.alphas_normal,
        self.alphas_reduce,
      ]
    else:
      self._arch_parameters = [
        self.alphas_normal
      ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

