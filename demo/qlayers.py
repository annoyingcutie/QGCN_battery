

import torchquantum as tq
#import torchquantum.layers
class QLayer_block1(tq.QuantumModule):
    def __init__(self,n_wires =4):
        super().__init__()
        self.wires = n_wires
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.wires,has_params=True,trainable=True)
        # self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.wires,has_params=True,trainable=True)
        # self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.wires,has_params=True,trainable=True)

        # self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.wires,has_params=True,trainable=True)
        # self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.wires,has_params=True,trainable=True)
        self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.wires,has_params=True,trainable=True)

        # self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=self.wires,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=self.wires,has_params=True,trainable=True,circular =True)
        # self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.wires,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.RYs1(self.q_device)
        # self.CRYs1(self.q_device)
        # self.RYs2(self.q_device)

        # self.RXs1(self.q_device)
        self.CRXs1(self.q_device)
        # self.RXs2(self.q_device)

        # self.RZs1(self.q_device)
        # self.CRZs1(self.q_device)
        self.RZs2(self.q_device)

        # self.CRZs3(self.q_device)
        # self.RYs3(self.q_device)
        # self.RXs3(self.q_device)
        
        # self.hadmard(self.q_device)

class QLayer_block2(tq.QuantumModule):
    def __init__(self,n_wires =4):
        super().__init__()
        self.wires = n_wires
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.wires,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.wires,has_params=True,trainable=True)
        # self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.wires,has_params=True,trainable=True)

        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.wires,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.wires,has_params=True,trainable=True)
        # self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.wires,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=self.wires,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=self.wires,has_params=True,trainable=True,circular =True)
        # self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.wires,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RYs2(self.q_device)

        self.RXs1(self.q_device)
        self.CRXs1(self.q_device)
        self.RXs2(self.q_device)

        # self.RZs1(self.q_device)
        # self.CRZs1(self.q_device)
        # self.RZs2(self.q_device)

        # self.CRZs3(self.q_device)
        # self.RYs3(self.q_device)
        # self.RXs3(self.q_device)
        
        # self.hadmard(self.q_device)

class QLayer_block3(tq.QuantumModule):
    def __init__(self,n_wires =4):
        super().__init__()
        self.wires = n_wires
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.wires,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.wires,has_params=True,trainable=True)
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.wires,has_params=True,trainable=True)

        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.wires,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.wires,has_params=True,trainable=True)
        self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.wires,has_params=True,trainable=True)

        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=self.wires,has_params=True,trainable=True,circular =True) #Op2QAllLayer
        self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=self.wires,has_params=True,trainable=True,circular =True)
        self.CRZs1 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=self.wires,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
        self.RYs2(self.q_device)

        self.RXs1(self.q_device)
        self.CRXs1(self.q_device)
        self.RXs2(self.q_device)

        self.RZs1(self.q_device)
        self.CRZs1(self.q_device)
        self.RZs2(self.q_device)

        # self.CRZs3(self.q_device)
        # self.RYs3(self.q_device)
        # self.RXs3(self.q_device)
        
        # self.hadmard(self.q_device)
class QLayer_block4(tq.QuantumModule):
    def __init__(self,n_wires =4):
        super().__init__()
        self.wires = n_wires
        self.layer_indexs  = dict()
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.wires,has_params=True,trainable=True)
        self.CRYs1 = tq.Op2QAllLayer(op=tq.CRY,n_wires=self.wires,has_params=True,trainable=True,circular =True) #Op2 self.CRXs1 = tq.Op2QAllLayer(op=tq.CRX,n_wires=self.wires,has_params=True,trainable=True,circular =True)
        
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        # self.hadmard(self.q_device)
        # add dense trainable gates
        self.RYs1(self.q_device)
        self.CRYs1(self.q_device)
       