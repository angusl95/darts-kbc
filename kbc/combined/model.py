import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from operations import *
from torch.autograd import Variable
from utils import drop_path

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


class Cell(nn.Module):

  def __init__(self, genotype,  C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()

    # if reduction_prev:
    #   self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    # else:
    #   self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    # self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    #s0 = self.preprocess0(s0)
    #s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class NetworkKBC(KBCModel):

  def __init__(self, C, num_classes, layers, criterion, regularizer, 
    genotype, interleaved, sizes: Tuple[int, int, int], emb_dim: int, 
    init_size: float = 1e-3, 
    reduction_flag = True, steps=4, multiplier=4, stem_multiplier=3):
    #TODO: remove stem multiplier from args?
    super(NetworkKBC, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._regularizer = regularizer
    self._steps = steps
    self._multiplier = multiplier
    self._stem_multiplier = stem_multiplier
    self.emb_dim = emb_dim
    if self.emb_dim % 20 != 0:
      raise ValueError('embedding size must be divisble by 20')
    self.emb_height = self.emb_dim//20
    self.sizes = sizes
    self._init_size = init_size
    self._interleaved = interleaved
    self._reduction_flag = reduction_flag
    self.embeddings = nn.ModuleList([
            nn.Embedding(s, emb_dim, sparse=True)
            for s in sizes[:2]
        ])
    self.embeddings[0].weight.data *= init_size
    self.embeddings[1].weight.data *= init_size

    C_curr = C
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
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      # if i == 2*layers//3:
      #   C_to_auxiliary = C_prev

    self.input_drop = torch.nn.Dropout(p=0.2)
    self.input_bn = torch.nn.BatchNorm2d(1)
    #self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.projection = nn.Linear(2*self.emb_dim*C_prev, self.emb_dim)#, bias=False)
    #self.classifier = nn.Linear(C_prev, num_classes)
    self.output_bn = nn.BatchNorm1d(self.emb_dim)
    self.output_drop = torch.nn.Dropout(p=0.3)

  def preprocess(self, lhs, rel):
    if self._interleaved:
      lhs = lhs.view([lhs.size(0),1,self.emb_height,20])
      rel = rel.view([rel.size(0),1,self.emb_height,20])
      s0 = torch.cat([lhs,rel],3)
      s0 = s0.view([lhs.size(0),1,2*self.emb_height,20])
    else:
      lhs = lhs.view([lhs.size(0),1,self.emb_height,20])
      rel = rel.view([rel.size(0),1,self.emb_height,20])
      s0 = torch.cat([lhs,rel], 2)
    s0 = self.input_bn(s0)
    s0 = self.input_drop(s0)
    s0 = s0.expand(-1,self._C, -1, -1)
    s1 = s0
    return s0, s1

  def score(self, x):
    lhs = self.embeddings[0](x[:, 0])
    rel = self.embeddings[1](x[:, 1])
    rhs = self.embeddings[0](x[:, 2])
    to_score = self.embeddings[0].weight

    # if self._interleaved:
    #   lhs = lhs.view([lhs.size(0),1,self.emb_height,20])
    #   rel = rel.view([rel.size(0),1,self.emb_height,20])
    #   s0 = torch.cat([lhs,rel],3)
    #   s0 = s0.view([lhs.size(0),1,2*self.emb_height,20])
    # else:
    #   lhs = lhs.view([lhs.size(0),1,self.emb_height,20])
    #   rel = rel.view([rel.size(0),1,self.emb_height,20])
    #   s0 = torch.cat([lhs,rel], 2)
    # s0 = self.input_bn(s0)
    # s0 = self.input_drop(s0)
    # s0 = s0.expand(-1,self._C, -1, -1)
    # s1 = s0

    s0, s1 = self.preprocess(lhs, rel)

    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

    #out = self.global_pooling(s1)
    out = s1.view(s1.size(0),1,-1)
    out = self.projection(out)
    out = out.squeeze()
    out = self.output_drop(out)
    out = self.output_bn(out)
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

    # if self._interleaved:
    #   lhs = lhs.view([lhs.size(0),1,self.emb_height,20])
    #   rel = rel.view([rel.size(0),1,self.emb_height,20])
    #   s0 = torch.cat([lhs,rel],3)
    #   s0 = s0.view([lhs.size(0),1,2*self.emb_height,20])
    # else:
    #   lhs = lhs.view([lhs.size(0),1,self.emb_height,20])
    #   rel = rel.view([rel.size(0),1,self.emb_height,20])
    #   s0 = torch.cat([lhs,rel], 2)
    # s0 = self.input_bn(s0)
    # s0 = self.input_drop(s0)
    # s0 = s0.expand(-1,self._C, -1, -1)
    # s1 = s0

    s0, s1 = self.preprocess(lhs,rel)

    for i, cell in enumerate(self.cells):
        s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
    
    out = s1.view(s1.size(0),1,-1)
    out = self.projection(out)
    out = out.squeeze()
    out = self.output_drop(out)
    out = self.output_bn(out)
    out = F.relu(out)
    out = out @ to_score.transpose(0,1)
    return (out), (lhs,rel,rhs)

  def get_rhs(self, chunk_begin: int, chunk_size: int):
    return self.embeddings[0].weight.data[
        chunk_begin:chunk_begin + chunk_size
    ].transpose(0, 1)

  def get_queries(self, queries: torch.Tensor):
    lhs = self.embeddings[0](queries[:, 0])
    rel = self.embeddings[1](queries[:, 1])
    # if self._interleaved:
    #   lhs = lhs.view([lhs.size(0),1,self.emb_height,20])
    #   rel = rel.view([rel.size(0),1,self.emb_height,20])
    #   s0 = torch.cat([lhs,rel],3)
    #   s0 = s0.view([lhs.size(0),1,2*self.emb_height,20])
    # else:
    #   lhs = lhs.view([lhs.size(0),1,self.emb_height,20])
    #   rel = rel.view([rel.size(0),1,self.emb_height,20])
    #   s0 = torch.cat([lhs,rel], 2)
    # s0 = self.input_bn(s0)
    # s0 = self.input_drop(s0)
    # s0 = s0.expand(-1,self._C, -1, -1)
    # s1 = s0

    s0, s1 = self.preprocess(lhs,rel)

    for i, cell in enumerate(self.cells):
      #print('cell', i, 'shapes of s0 and s1:', s0.shape, s1.shape)
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
    out = s1.view(s1.size(0),1,-1)
    out = self.projection(out)
    out = out.squeeze()
    out = self.output_drop(out)
    out = self.output_bn(out)
    out = F.relu(out)

    return out


