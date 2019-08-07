from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')# reduce reduce_concat')

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
    'dil_conv_5x5',
    'conv_7x1_1x7'
    #'linear'
]

ConvE = Genotype(
    normal = [
    ('conv_3x3', 0),
    ('identity', 1),
    ('identity', 2),
    ('identity', 3)
    ],
  normal_concat = [3]
)

DARTSNet = Genotype(normal=[('identity', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('conv_7x7', 3)], normal_concat=[4])