from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5',
#     'conv_3x3',
#     'conv_5x5',
#     'conv_7x7'
# ]

# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'conv_3x3',
#     'conv_5x5',
#     'conv_7x7'
# ]

PRIMITIVES = [
    'none',
    'identity',
    'relu',
    'tanh',
    'sigmoid']

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('skip_connect', 0), 
    ('sep_conv_3x3', 1), 
    ('skip_connect', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[
    ('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 1), 
    ('skip_connect', 0), 
    ('skip_connect', 0), 
    ('dil_conv_3x3', 2)
    ], normal_concat=[2, 3, 4, 5], 
    reduce=[
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 1), 
    ('skip_connect', 2), 
    ('max_pool_3x3', 1), 
    ('max_pool_3x3', 0), 
    ('skip_connect', 2), 
    ('skip_connect', 2), 
    ('max_pool_3x3', 1)], 
    reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

KBCNet = Genotype(normal=[('relu', 1), 
    ('tanh', 0), 
    ('tanh', 0), 
    ('relu', 1), 
    ('identity', 0), 
    ('relu', 1), 
    ('tanh', 0), 
    ('identity', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])


TestNet = Genotype(normal=[('none', 1), ('none', 0), ('none', 1), ('none', 0), ('conv_5x5', 2), ('conv_7x7', 0), ('conv_7x7', 4), ('conv_7x7', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('conv_5x5', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))

ConvE = Genotype(
    normal = [
        ('conv_3x3', 0),
    ('zero', 1),
    ('zero', 0),
    ('zero', 1),
    ('zero', 0),
    ('zero', 1),
    ('zero', 0),
    ('zero', 1)],
    normal_concat = [2,3,4,5],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
    )

DARTSNet_V1 = Genotype(normal=[('tanh', 1), ('none', 0), ('relu', 1), ('relu', 2), ('tanh', 2), ('tanh', 1), ('relu', 1), ('relu', 2)], normal_concat=range(2, 6), reduce=[('relu', 0), ('none', 1), ('identity', 0), ('tanh', 2), ('identity', 1), ('relu', 3), ('relu', 1), ('relu', 3)], reduce_concat=range(2, 6))
DARTSNet_V2 = Genotype(normal=[('relu', 1), ('relu', 0), ('relu', 1), ('identity', 2), ('relu', 3), ('relu', 1), ('tanh', 2), ('relu', 1), ('relu', 1), ('tanh', 2)], normal_concat=range(3, 7), reduce=[('relu', 1), ('identity', 0), ('tanh', 1), ('identity', 0), ('identity', 0), ('identity', 2), ('tanh', 2), ('identity', 3), ('tanh', 1), ('relu', 4)], reduce_concat=range(3, 7))

Vis = Genotype(normal=[('identity', 1), ('relu', 0), ('identity', 1), ('identity', 2), ('identity', 1), ('identity', 2), ('identity', 1), ('identity', 2), ('identity', 3), ('identity', 2)], normal_concat=range(3, 7), reduce=[('identity', 1), ('relu', 0), ('tanh', 2), ('tanh', 0), ('tanh', 3), ('tanh', 0), ('relu', 4), ('relu', 0), ('identity', 2), ('identity', 3)], reduce_concat=range(3, 7))

