import torch
import torchquantum as tq
import numpy as np
from torchquantum.measurement import expval
from typing import Union, List


# def expval(q_device: tq.QuantumDevice,
#            wires: Union[int, List[int]],
#            observables: Union[tq.Observable, List[tq.Observable]],
#            mag = False):

#     all_dims = np.arange(q_device.states.dim())
#     if isinstance(wires, int):
#         wires = [wires]
#         observables = [observables]

#     # rotation to the desired basis
#     for wire, observable in zip(wires, observables):
#         for rotation in observable.diagonalizing_gates():
#             rotation(q_device, wires=wire)

#     states = q_device.states
#     # compute magnitude
#     state_mag = torch.abs(states) ** 2
#     if mag:
#         return state_mag

#     expectations = []
#     for wire, observable in zip(wires, observables):
#         # compute marginal magnitude
#         reduction_dims = np.delete(all_dims, [0, wire + 1])
#         probs = state_mag.sum(list(reduction_dims))
#         res = probs.mv(observable.eigvals.real.to(probs.device))
#         expectations.append(res)

#     return torch.stack(expectations, dim=-1)


class Measure(tq.QuantumModule):
    def __init__(self, obs,wires , v_c_reg_mapping=None):
        super().__init__()
        self.obs = obs
        self.v_c_reg_mapping = v_c_reg_mapping
        self.wires = wires

    def forward(self, q_device: tq.QuantumDevice,mag=False):
        self.q_device = q_device
        x = expval(q_device, self.wires, [self.obs()] *
                   len(self.wires))#,mag)

        if self.v_c_reg_mapping is not None:
            c2v_mapping = self.v_c_reg_mapping['c2v']
            """
            the measurement is not normal order, need permutation 
            """
            perm = []
            for k in range(x.shape[-1]):
                if k in c2v_mapping.keys():
                    perm.append(c2v_mapping[k])
            x = x[:, perm]

        if self.noise_model_tq is not None and \
                self.noise_model_tq.is_add_noise:
            return self.noise_model_tq.apply_readout_error(x)
        else:
            return x

    def set_v_c_reg_mapping(self, mapping):
        self.v_c_reg_mapping = mapping
