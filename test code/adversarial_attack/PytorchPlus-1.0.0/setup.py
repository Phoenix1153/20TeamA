from distutils.core import setup

setup(
    name='PytorchPlus',
    version='1.0.0',
    package_dir = {'PytorchPlus': 'PytorchPlus'},
    packages=['PytorchPlus.adversarial_attack', 'PytorchPlus.adversarial_attack.CW_attack_class',

              'PytorchPlus.QNNs',
              'PytorchPlus.QNNs.bnn_res20_1w1a', 'PytorchPlus.QNNs.bnn_res20_1w1a.modules',
              'PytorchPlus.QNNs.bwn_res20_1w32a', 'PytorchPlus.QNNs.bwn_res20_1w32a.modules',
              'PytorchPlus.QNNs.horq_res20_1w2a', 'PytorchPlus.QNNs.horq_res20_1w2a.modules',
              'PytorchPlus.QNNs.irnet_res20_1w1a', 'PytorchPlus.QNNs.irnet_res20_1w1a.modules',
              'PytorchPlus.QNNs.xnor_res20_1w1a', 'PytorchPlus.QNNs.xnor_res20_1w1a.modules',

              'PytorchPlus.RC',

              'PytorchPlus.active_learning', 'PytorchPlus.active_learning.utils',
              ],
    author='teamA',
    author_email='chongzhizhang@buaa.edu.cn',
    description='PytorchPlus module',
)
'''packages =['PytorchPlus.adversarial_attack', 'PytorchPlus.adversarial_attack.CW_attack_class',

           'PytorchPlus.QNNs',
           'PytorchPlus.QNNs.bnn_res20_1w1a','PytorchPlus.QNNs.bnn_res20_1w1a.modules',
           'PytorchPlus.QNNs.bwn_res20_1w32a','PytorchPlus.QNNs.bwn_res20_1w32a.modules',
           'PytorchPlus.QNNs.horq_res20_1w2a','PytorchPlus.QNNs.horq_res20_1w2a.modules',
           'PytorchPlus.QNNs.irnet_res20_1w1a','PytorchPlus.QNNs.irnet_res20_1w1a.modules',
           'PytorchPlus.QNNs.xnor_res20_1w1a','PytorchPlus.QNNs.xnor_res20_1w1a.modules',

           'PytorchPlus.RC',

           'PytorchPlus.active_learning','PytorchPlus.active_learning.utils',
           ],'''