import torch
from torch import nn
from torch.autograd import Variable
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset


features, labels = load_digits(return_X_y=True)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,shuffle=True, random_state=3)

print ("Data Shapes")
print ("====================")
print ("shape of features:", features.shape)
print ("shape of labels: ", labels.shape)



# Model
class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, hidden)
        self.l5 = nn.Linear(hidden, output)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.l4(out))
        out = F.log_softmax(self.l5(out), dim=1)
        return out


# parameters
input = features.shape[1]
output = 10
hidden = 100

net = Net(input, hidden, output)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
criterion = nn.NLLLoss()
epochs=10

print ()
print ("Model Structure")
print ("=====================")
print (net)


train_dataset = TensorDataset(torch.from_numpy(features_train), torch.from_numpy(labels_train))
test_dataset = TensorDataset(torch.from_numpy(features_test), torch.from_numpy(labels_test))

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=32)


def train():
    net.train()
    losses = []
    for x, y in train_loader:
        x = Variable(x).float()
        y = Variable(y).long()
        y_pred = net(x)
        loss = criterion(y_pred, y)
        print ("loss: ", loss.item())
        losses.append(loss.item()) # Individual loss after each example in the batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Training
for e in range(2):
    train() # Train for 2 epochs

# Saving model into coreml model
from onnx_coreml import convert
import onnx

dummy = Variable(torch.FloatTensor(32, 64))
torch.onnx.export(net, dummy, 'mnist-model.proto', verbose=True)
model = onnx.load('mnist-model.proto')
coreml_model = convert(
    model,
    'classifier',
    image_input_names=['features'],
    image_output_names=['labels'],
    class_labels=[i for i in range(9)],
)
coreml_model.save('mnist-model.mlmodel')





    
