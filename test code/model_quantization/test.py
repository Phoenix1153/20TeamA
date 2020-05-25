# test import PytorchPlus
import PytorchPlus

# test import PytorchPlus.QNNs
import PytorchPlus.QNNs

# test import PytorchPlus.QNNs.bnn_res20_1w1a
import PytorchPlus.QNNs.bnn_res20_1w1a as bnn

# test import PytorchPlus.QNNs.xnor_res20_1w1a
import PytorchPlus.QNNs.xnor_res20_1w1a as xnor

# test import PytorchPlus.QNNs.bwn_res20_1w32a
import PytorchPlus.QNNs.bwn_res20_1w32a as bwn

# test import PytorchPlus.QNNs.horq_res20_1w2a
import PytorchPlus.QNNs.horq_res20_1w2a as horq

# test import PytorchPlus.QNNs.irnet_res20_1w1a
import PytorchPlus.QNNs.irnet_res20_1w1a as irnet

# test loading / training / inference / interrupted of bnn
bnn.__main__.main()

# test loading / training / inference / interrupted of bwn
bwn.__main__.main()

# test loading / training / inference / interrupted of horq
horq.__main__.main()

# test loading / training / inference / interrupted of irnet
irnet.__main__.main()

# test loading / training / inference / interrupted of xnor
xnor.__main__.main()

