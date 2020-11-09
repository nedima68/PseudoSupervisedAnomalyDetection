
from .CNN_SVDD32x32 import DefectDetectCNN32x32, DefectDetectCNN32x32_Autoencoder
from .CNN_SVDD64x64 import DefectDetectCNN64x64, DefectDetectCNN64x64_Autoencoder
from .MNIST_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder



def build_network(net_name, rep_dim = 128, channel_num = 3):
    """Builds the neural network."""

    implemented_networks = ( 'CNN32x32Deep',  'DefectDetectCNN32x32', 'DefectDetectCNN64x64', 'MNIST_LeNet')
    assert net_name in implemented_networks

    net = None

   
    if net_name == 'CNN32x32Deep':
        net = CNN32x32Deep()

    if net_name == 'DefectDetectCNN32x32':
        net = DefectDetectCNN32x32(rep_dim, channel_num)

    if net_name == 'DefectDetectCNN64x64':
        net = DefectDetectCNN64x64()

    if net_name == 'MNIST_LeNet':
        net = MNIST_LeNet()

       
    return net


def build_autoencoder(net_name, rep_dim = 128, channel_num = 3):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ( 'CNN32x32Deep', 'DefectDetectCNN32x32', 'DefectDetectCNN64x64', 'MNIST_LeNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'CNN32x32Deep':
        ae_net = CNN32x32Deep_Autoencoder()

    if net_name == 'DefectDetectCNN32x32':
        ae_net = DefectDetectCNN32x32_Autoencoder(rep_dim, channel_num)

    if net_name == 'DefectDetectCNN64x64':
        ae_net = DefectDetectCNN64x64_Autoencoder()

    if net_name == 'MNIST_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    return ae_net
