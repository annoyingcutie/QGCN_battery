from turtle import forward
# from cirq import givens
import numpy as np
import torch
import math
#from qiskit.providers import backend
#from qiskit import *
import math
#from qiskit.test.mock import FakeVigo
#from qiskit.providers.aer import AerSimulator
import sys
import copy

def genGivens2q(param,p,q):
    # print('givens:',p,q)
    # print(param)
    circuit = QuantumCircuit(2)
    if p == 0 and q== 1 :
        circuit.x(1)
        circuit.cry(param,1,0)
        circuit.x(1)
    elif p == 1 and q== 0 :
        circuit.x(1)
        circuit.cx(1,0)
        circuit.cry(param,1,0)
        circuit.cx(1,0)
        circuit.x(1)
    elif p==0 and q==2:
        circuit.x(0)
        circuit.cry(param,0,1)
        circuit.x(0)
    elif p==2 and q==0:
        circuit.x(0)
        circuit.cx(0,1)
        circuit.cry(param,0,1)
        circuit.cx(0,1)
        circuit.x(0)
    elif p==0 and q==3:
        #化简
        circuit.x(1)
        # circuit.swap(0,1)
        circuit.cx(0,1)
        circuit.cry(param,1,0)
        circuit.cx(0,1)
        # circuit.swap(0,1)
        circuit.x(1)
    elif p==3 and q==0:
        circuit.x(1)
        circuit.cx(1,0)
        circuit.cry(param,0,1)
        circuit.cx(1,0)
        circuit.x(1)
    elif p==1 and q==2:
        circuit.cx(1,0)
        circuit.cry(param,0,1)
        circuit.cx(1,0)
    elif p==2 and q==1:
        circuit.cx(1,0)
        circuit.cx(0,1)
        circuit.cry(param,0,1)
        circuit.cx(0,1)
        circuit.cx(1,0)
    elif p==1 and q==3:
        circuit.cry(param,0,1)
    elif p==3 and q==1:
        circuit.cx(0,1)
        circuit.cry(param,0,1)
        circuit.cx(0,1)
    elif p==2 and q==3:
        circuit.cry(param,1,0)
    elif p==3 and q==2:
        circuit.cx(1,0)
        circuit.cry(param,1,0)
        circuit.cx(1,0)
    return circuit

def change2q(p,q):
    # if p>q:
    #     t=p
    #     p=q
    #     q=t
    # print('swap:',p,q)
    circuit = QuantumCircuit(2)
    if (p == 0 and q== 1) or (q == 0 and p== 1):
        circuit.x(1)
        circuit.cx(1,0)
        circuit.x(1)
    elif (p==0 and q==2) or (q == 0 and p== 2):
        circuit.x(0)
        circuit.cx(0,1)
        circuit.x(0)
    elif (p==0 and q==3) or (q == 0 and p== 3):
        circuit.x(1)
        circuit.swap(0,1)
        circuit.x(1)
    elif (p==1 and q==2) or (q == 1 and p== 2):
        circuit.swap(0,1)
    elif (p==1 and q==3) or (q == 1 and p== 3):
        circuit.cx(0,1)
    elif (p==2 and q==3) or (q == 2 and p== 3):
        circuit.cx(1,0)
    return circuit

# def change2q_inv(p,q):
#     if p>q:
#         t=p
#         p=q
#         q=t
#     print('swap:',p,q)
#     circuit = QuantumCircuit(2)
#     if p == 0 and q== 1 :
#         circuit.x(1)
#         circuit.cx(1,0)
#         circuit.x(1)
#     elif p==0 and q==2:
#         circuit.x(0)
#         circuit.cx(0,1)
#         circuit.x(0)
#     elif p==0 and q==3:
#         circuit.x(1)
#         circuit.swap(0,1)
#         circuit.x(1)
#     elif p==1 and q==2:
#         circuit.swap(0,1)
#     elif p==1 and q==3:
#         circuit.cx(0,1)
#     elif p==2 and q==3:
#         circuit.cx(1,0)
#     return circuit


def get_control_info(nq,i,plist):
    qubit_list = list(range(nq))
    pc_list = copy.deepcopy(plist) 
    q1 =nq-i-2
    q2 =nq-i-1
    qubit_list.pop(q1)
    qubit_list.pop(q1)
    qubit_list.append(q1)
    qubit_list.append(q2)
    pc_list.reverse()
    pc_list.pop(q1)
    pc_list.pop(q1)
    pc_list.reverse()
    state = ''.join([str(elem) for elem in pc_list])
    # print(state)
    return qubit_list,state

class GivensRotations(torch.nn.Module):
    def __init__(self,n_qubits=3,pqs=[]):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_dims =int(math.pow(2,n_qubits))
        self.pqs = pqs
        self.n_edges = len(pqs)
        self.build(pqs)
        self.trans =None
        self.circuit =None

    def build(self,pqs):
        num = len(pqs)
        print('num:',num)
        self.params = torch.nn.ParameterList()
        # self.circuits =[]

        for i in range(num):
            para = torch.nn.Parameter(torch.randn(1),requires_grad=True)
            self.params.append(para)

    def build_circuit(self,pqs):
        self.circuit =None
        num = len(pqs)
        print('num:',num)
        self.circuits =[]
        for i in range(num):
            circuit = self.genCircuit(math.pi/3,pqs[i][0],pqs[i][1])
            self.circuits.append(circuit)
        self.circuit = QuantumCircuit(self.n_qubits)
        # self.circuit.h([0,1])
        for circ in self.circuits:
            self.circuit =  self.circuit +circ


    def genMat(self,param,p,q):
        mat = torch.eye(self.n_dims)
        c = torch.cos(param/2)
        s = torch.sin(param/2)
        mat[p,p]=c
        mat[p,q]=-s
        mat[q,p]=s
        mat[q,q]=c
        return mat

    # def getMatAndState(self):
    #     mats = []
    #     state = torch.ones(self.n_dims,1)/math.sqrt(self.n_dims)
    #     # state = torch.zeros(self.n_dims,1)
    #     # state[0,0]=1
    #     for i in range(self.n_edges):
    #         mat = self.genMat(self.params[0],self.pqs[i][0],self.pqs[i][1])
    #         state = torch.mm(mat,state)
    #         mats.append(mat)
    #     state =  abs(state)
    #     return mats,state

    def genCircuit(self,param,p,q):
        # param = param.item()
        qformat = '{0:0'+str(self.n_qubits)+'b}'
        pbit =qformat.format(p)
        plist =list(pbit)
        qbit = qformat.format(q)
        qlist =list(qbit)
        nq =self.n_qubits
        # print(pbit,qbit)
        change_list =[]
        for i in range(self.n_qubits-2):
            if plist[i] == qlist[i]:
                continue
            # print(i,' sub state:',plist[i:i+2],qlist[i:i+2])
            pp = int(''.join([str(elem) for elem in plist[i:i+2]]),2)
            qq = int(''.join([str(elem) for elem in qlist[i:i+2]]),2)
            qubit_list,state = get_control_info(nq,i,plist)
            change_list.append([i,pp,qq,qubit_list,state])

            plist[i:i+2] = qlist[i:i+2]
        i = self.n_qubits-2
        if plist[i:self.n_qubits] != qlist[i:self.n_qubits]:
            pp = int(''.join([str(elem) for elem in plist[i:i+2]]),2)
            qq = int(''.join([str(elem) for elem in qlist[i:i+2]]),2)
            qubit_list,state = get_control_info(nq,i,plist)
            change_list.append([i,pp,qq,qubit_list,state])

        # print('first circuit:',change_list)

        circuit = QuantumCircuit(self.n_qubits)

        for sw in change_list[:-1]:
            qubit_list = sw[3]
            state = sw[4]
            circ = change2q(sw[1],sw[2])
            # print(circ)
            gate = circ.to_gate().control(num_ctrl_qubits= self.n_qubits-2,  ctrl_state=state,)
            circuit.append(gate,qubit_list)


        sw =change_list[-1]
        # print('shift circuit:',sw)
        qubit_list = sw[3]
        state = sw[4]
        circ = genGivens2q(param,sw[1],sw[2])
        # print(circ)
        gate = circ.to_gate().control(num_ctrl_qubits= self.n_qubits-2,  ctrl_state=state)
        circuit.append(gate,qubit_list)


        change_list = change_list[::-1]
        # print('inverse circuit:',change_list)

        for sw in change_list[1:]:
            qubit_list = sw[3]
            state = sw[4]
            circ = change2q(sw[1],sw[2])
            gate = circ.to_gate().control(num_ctrl_qubits= self.n_qubits-2,  ctrl_state=state)
            circuit.append(gate,qubit_list)

        return circuit
    
    def getCircuit(self):
        if self.trans ==None:
            self.build_circuit(self.pqs)
        return self.circuit

    def genTrans(self):
        self.trans = torch.eye(self.n_dims)
        for i in range(self.n_edges):
            mat = self.genMat(torch.tensor(math.pi/3),self.pqs[i][0],self.pqs[i][1])#torch.tensor(math.pi/3)
            self.trans = torch.mm(mat,self.trans)
    
    def get_trans(self):
        if self.trans ==None:
            self.genTrans()
        return self.trans

    def forward(self,h):
        if self.trans ==None:
            self.genTrans()
        h =  torch.mm(self.trans,h)
        return h
##test
# gr = GivensRotations(4)
# circuit = QuantumCircuit(4)
# circuit.h([0,1,2,3])
# qb1 = 4
# qb2 = 11
# val = math.pi/7
# circ = gr.genCircuit(val,qb1,qb2)
# circuit = circuit +circ
# print(circuit)
# # print(circuit.decompose().decompose())

# simulator = AerSimulator(method='matrix_product_state')

# # Execute and get saved data
# tcirc = transpile(circuit, simulator)
# global_phase = tcirc.global_phase
# print(global_phase)
# tcirc.save_statevector(label='my_sv')
# result = simulator.run(tcirc).result()
# data = result.data(0)
# final_state = data['my_sv']
# final_state =  abs(np.array(final_state))
# print('final_state:\n',final_state)
# state = torch.ones(16,1)/4
# mat = gr.genMat(torch.tensor(val),qb1,qb2)
# state = torch.mm(mat,state)
# state = abs(state)
# print('final_state:\n',state)
# sys.exit(0)

# class GivensRotations2(torch.nn.Module):
#     def __init__(self,n_qubits=2,pqs=[]):
#         super().__init__()
#         self.n_qubits = n_qubits
#         self.n_dims =int(math.pow(2,n_qubits))
#         self.build(pqs)
#         self.pqs = pqs
#         self.n_edges = len(pqs)

#     def build(self,pqs):
#         num = len(pqs)
#         print('num:',num)
#         self.params = torch.nn.ParameterList()
#         self.circuits =[]

#         for i in range(num):
#             para = torch.nn.Parameter(torch.randn(1),requires_grad=True)
#             self.params.append(para)

#             circuit = self.genCircuit(self.params[i],pqs[i][0],pqs[i][1])
#             self.circuits.append(circuit)
            
#         self.circuit = QuantumCircuit(self.n_qubits)
#         self.circuit.h([0,1])
#         for circ in self.circuits:
#             self.circuit =  self.circuit +circ
    
#     def getCircuit(self):
#         return self.circuit
#     def getMatAndState(self):
#         mats = []
#         state = torch.ones(self.n_dims,1)*0.5
#         # state = torch.zeros(self.n_dims,1)
#         # state[0,0]=1
#         for i in range(self.n_edges):
#             mat = self.genMat(self.params[i],self.pqs[i][0],self.pqs[i][1])
#             state = torch.mm(mat,state)
#             mats.append(mat)
#         if state[0]<0:
#             state = state*-1
#         return mats,state

#     def genMat(self,param,p,q):
#         mat = torch.eye(self.n_dims)
#         c = torch.cos(param/2)
#         s = torch.sin(param/2)
#         mat[p,p]=c
#         mat[p,q]=-s
#         mat[q,p]=s
#         mat[q,q]=c
#         return mat

#     def genCircuit(self,param,p,q):
#         param = param.item()
#         circ = genGivens2q(param,p,q)
#         return circ

#     def forward(self,h):
#         for i in range(self.n_edges):
#             mat = self.genMat(self.params[i],self.pqs[i][0],self.pqs[i][1])
#             h =  torch.mm(mat,h)
#         return h



