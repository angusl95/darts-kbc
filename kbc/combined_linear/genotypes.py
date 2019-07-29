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

PRIMITIVES = [
    'identity',
    'max_pool_3x3',
    'avg_pool_3x3',
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

KBCNet = Genotype(
  normal = [
    ('conv_3x3', 0),
    ('none', 1)
    # ('skip_connect', 0),
    # ('skip_connect', 2),
    # ('skip_connect', 0),
    # ('skip_connect', 3),
    # ('skip_connect', 1),
    # ('skip_connect', 1),
    # ('skip_connect', 0),
    # ('skip_connect', 1),
    ],
  #normal_concat = [4, 5, 6],
  normal_concat = [2],
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

TestNet = Genotype(
    normal = [
    ('conv_3x3', 0),
    ('skip_connect', 1),
    ('skip_connect', 2),
    ('skip_connect', 3)
    # ('skip_connect', 0),
    # ('skip_connect', 2),
    # ('skip_connect', 0),
    # ('skip_connect', 3),
    # ('skip_connect', 1),
    # ('skip_connect', 1),
    # ('skip_connect', 0),
    # ('skip_connect', 1),
    ],
  #normal_concat = [4, 5, 6],
  normal_concat = [3],
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





