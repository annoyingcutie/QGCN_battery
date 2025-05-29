from torch_geometric.data import Data, InMemoryDataset, DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchquantum as tq
import math
from torch_geometric.utils import to_scipy_sparse_matrix
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from givens_roatation import GivensRotations
import scipy.sparse as sp
from qlayers import *
from measure import *



class BatteryGraphDataset(InMemoryDataset):
    def __init__(self, data, labels, transform=None,points=(256,8)):
        """
        :param data: Tensor of shape [91, 8 * 256 * 4] (batteries, flattened nodes and features)
        :param labels: Tensor of shape [91, 1] (batteries, labels)
        """
        self.data_array = data
        self.label_array = labels
        super().__init__(None, transform)

        self.data_list = []
        num_batteries = data.size(0)
        num_train = int(0.8 * num_batteries)
        print("points =", points)
        print("type(point) =", type(points[0]))
        print("type(cycles_per_battery) =", type(points[1]))

        time_points, cycles_per_battery = points

        for i in range(num_batteries):
            # Reshape into [2048, 4] → [nodes, features]
            print(data[i].size())
            x = data[i].reshape((time_points*cycles_per_battery), 4)
            y = labels[i].repeat(x.size(0), 1)

            # Create masks
            train_mask = torch.full((x.size(0),), i < num_train, dtype=torch.bool)
            test_mask = ~train_mask

            edge_index = []

            # Inner-cycle: connect nodes sequentially within each cycle
            for cycle in range(cycles_per_battery):
                start = cycle * time_points
                for t in range(time_points - 1):
                    edge_index.append([start + t, start + t + 1])
                    edge_index.append([start + t + 1, start + t])

            # Cross-cycle: connect same time point across different cycles
            for t in range(time_points):
                for cycle in range(cycles_per_battery - 1):
                    current = cycle * time_points + t
                    next_cycle = (cycle + 1) * time_points + t
                    edge_index.append([current, next_cycle])
                    edge_index.append([next_cycle, current])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            graph = Data(
                x=x.float(),
                y=y.flatten().long(),
                edge_index=edge_index,
                train_mask=train_mask,
                test_mask=test_mask
            )
            self.data_list.append(graph)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def normalize_adj(mx):

    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

class GCN(torch.nn.Module):
    def __init__(self, nfeat,  nclass,edge_index):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nclass, bias= False)
        self.edge_index = edge_index

    def forward(self, x):
        # x, edge_index = data.x, data.edge_index # 取出x, edge_index
        x = self.conv1(x, self.edge_index)
        return x
    



class QGCN(tq.QuantumModule):
    def __init__(self, g, n_feature, n_vqc_wires=4):
        super().__init__()

        self.n_qubit = math.ceil(math.log(g.x.shape[0], 2))  # Determine required qubits

        # Quantum GCN Layer (Using Edge Index Transpose as `pqs`)
        self.qgcn1 = GivensRotations(self.n_qubit, g.edge_index.T)

        # Linear layer for classification
        #self.linear1 = torch.nn.Linear(int(np.log2(n_feature)), 2)
        self.linear1 = torch.nn.Linear(int(4), 2)

        # Normalize adjacency matrix
        self.adj = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.x.shape[0])
        self.adj = normalize_adj(self.adj)
        self.adj = torch.tensor(self.adj.todense(), dtype=torch.float32)

        # Quantum layers
        self.n_wires = n_vqc_wires
        self.q_layers = nn.ModuleList([QLayer_block4(self.n_wires) for _ in range(13)])

        self.encoder = tq.StateEncoder()
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.measure = Measure(obs=tq.PauliZ, wires=range(self.n_wires))

    def forward(self, batch):
        h = batch.x  # Use batch-wise node features
        batch_size = h.shape[0]

        h = self.qgcn1(h)  # Apply Quantum GCN layer
        h = self.encoder(self.q_device, h)  # Quantum encoding

        for layer in self.q_layers:
            layer(self.q_device)

        h = self.measure(self.q_device)  # Quantum measurement
        h = self.linear1(h)  # Linear transformation

        return h

loss_func = torch.nn.CrossEntropyLoss()
#loss_func = torch.nn.BCELoss()

''''
num_class_0 = 24
num_class_1 = 67
total = num_class_0 + num_class_1

weight_0 = total / (2 * num_class_0)
weight_1 = total / (2 * num_class_1)
'''


def train(model, train_loader, epochs=20, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    #class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)
    #loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)  

            optimizer.zero_grad()
            output = model(batch)  

            # Ensure train_mask is correctly indexed
            loss = loss_func(output[batch.train_mask], batch.y[batch.train_mask])  
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")





@torch.no_grad()
def test(model, test_loader, device='cpu'):
    model.eval()
    count = 0
    correct = 0
    total = 0
    total_loss = 0
    #loss_func = torch.nn.CrossEntropyLoss()
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)

    for batch in test_loader:
        print(count)
        batch = batch.to(device)  # Move batch to the correct device
        output = model(batch)  # Forward pass

        loss = loss_func(output[batch.test_mask], batch.y[batch.test_mask])  # Compute loss

        total_loss += loss.item()

        preds = output[batch.test_mask].argmax(dim=1)  # Get predictions
        print(preds.numpy().tolist())
        correct += preds.eq(batch.y[batch.test_mask]).sum().item()  # Count correct predictions
        total += batch.test_mask.sum().item()  # Count total test samples
        count = count + 1

    accuracy = correct / total if total > 0 else 0  # Avoid division by zero
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.4f}")
    return accuracy



