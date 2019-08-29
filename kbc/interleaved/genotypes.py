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
    #'identity',
    'max_pool_3x3',
    'avg_pool_3x3',
    'conv_3x3',
    'conv_5x5',
    #'conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_1x5',
    'conv_5x1'
    #'conv_7x1_1x7'
    #'linear'
]

ConvE = Genotype(
    normal = [
    ('conv_3x3', 0)
    ],
  normal_concat = [0]
)

DARTSNet = Genotype(normal=[('identity', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('conv_7x7', 3)], normal_concat=[4])
WNNet_V1 = Genotype(normal=[('conv_5x5', 0), ('identity', 1), ('identity', 2), ('identity', 3), ('conv_7x7', 4)], normal_concat=[5])
FBNet_V1 = Genotype(normal=[('conv_3x3', 0), ('identity', 1), ('conv_5x5', 2), ('dil_conv_5x5', 3), ('conv_3x3', 4)], normal_concat=[5])

WNNet_V2 = Genotype(normal=[('conv_1x5', 0), ('identity', 1), ('dil_conv_5x5', 2), ('identity', 3), ('dil_conv_5x5', 4)], normal_concat=[5])
FBNet_V2 = Genotype(normal=[('identity', 0), ('identity', 1), ('dil_conv_5x5', 2), ('identity', 3), ('identity', 4)], normal_concat=[5])