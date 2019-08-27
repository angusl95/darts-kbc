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


KBCNet = Genotype(normal=[('relu', 1), 
    ('tanh', 0), 
    ('tanh', 0), 
    ('relu', 1), 
    ('identity', 0), 
    ('relu', 1), 
    ('tanh', 0), 
    ('identity', 2)], normal_concat=[2, 3, 4, 5])


TestNet = Genotype(normal=[('none', 1), ('none', 0), ('none', 1), ('none', 0), ('conv_5x5', 2), ('conv_7x7', 0), ('conv_7x7', 4), ('conv_7x7', 2)], normal_concat=range(2, 6))

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
    normal_concat = [2,3,4,5]
    )

DARTSNet_V1 = Genotype(normal=[('tanh', 1), ('none', 0), ('relu', 1), ('relu', 2), ('tanh', 2), ('tanh', 1), ('relu', 1), ('relu', 2)], normal_concat=range(2, 6))
DARTSNet_V2 = Genotype(normal=[('relu', 1), ('relu', 0), ('relu', 1), ('identity', 2), ('relu', 3), ('relu', 1), ('tanh', 2), ('relu', 1), ('relu', 1), ('tanh', 2)], normal_concat=range(3, 7))

WNNet_V1 = Genotype(normal=[('identity', 1), ('tanh', 0), ('identity', 1), ('identity', 2), ('identity', 1), ('relu', 3), ('relu', 2), ('tanh', 1), ('relu', 1), ('identity', 4)], normal_concat=range(3, 7))
FBNet_V1 = Genotype(normal=[('identity', 1), ('relu', 0), ('identity', 1), ('identity', 2), ('identity', 1), ('identity', 2), ('identity', 1), ('identity', 2), ('identity', 3), ('identity', 2)], normal_concat=range(3, 7))

Vis = Genotype(normal=[('tanh', 1), ('tanh', 0), ('identity', 2), ('tanh', 1), ('identity', 2), ('tanh', 1), ('identity', 2), ('tanh', 1), ('identity', 2), ('tanh', 1)], normal_concat=range(3, 7))