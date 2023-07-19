import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_emb, nhead: int = 2, dim_feedforward: int = 60, dropout: float =0.1, activation = nn.GELU):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attn = nn.MultiheadAttention(dim_emb, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_emb, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_emb)

        self.norm1 = nn.LayerNorm(dim_emb)
        self.norm2 = nn.LayerNorm(dim_emb)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, data, src_mask=None, src_key_padding_mask=None):
        
        # MultiHeadAttention
        x = self.attn(data, data, data, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        
        # add & norm
        x = data + self.dropout1(x)
        x = self.norm1(x)
        
        # Implementation of Feedforward model
        x1 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # add & norm
        x = x + self.dropout2(x1)
        x = self.norm2(x)
        
        return x        

class MLP(nn.Module):
    def __init__(self, dim_emb, dim_feedforward: int = 10, number_classes: int = 92, dropout: float =0.1, activation = nn.GELU):
        super(MLP, self).__init__()
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_emb, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, number_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))    
            
        return x

class Net(nn.Module):
    def __init__(self, dim_emb, dim_emb_transformer: int = 64, number_classes: int = 4, dropout: float =0.1, activation = nn.GELU, nhead: int = 2, dim_feedforward: int = 60):
        super(Net, self).__init__()
    
        # self.conv2d = nn.Conv2d(1, dim_emb, kernel_size=(1, 3), stride=(1, 2))
        self.lin = torch.nn.Linear(dim_emb, dim_emb_transformer)
        # self.transform = MLP(dim_emb, number_classes=dim_emb_transformer)
        self.te = TransformerEncoderLayer(dim_emb_transformer, nhead, dim_feedforward, dropout, activation)
        self.mlp = MLP(dim_emb_transformer, number_classes=number_classes)
        self.activation = nn.GELU()
        
    def forward(self, data):
        # x = self.conv2d(data)
        # x = self.transform(data)
        x = self.lin(data)
        x = self.te(x)
        x = self.mlp(x[:,0])
        x = self.activation(x)
        return x
    

class Netv2(nn.Module):
    def __init__(self, dim_emb, dim_emb_transformer: int = 64, number_classes: int = 4, dropout: float =0.1, activation = nn.GELU, nhead: int = 2, dim_feedforward: int = 60):
        super(Netv2, self).__init__()
        self.depthwiseconv = nn.Conv2d(in_channels=307, out_channels=1, kernel_size=(307,1), stride=1, dilation=1)
        self.adaptivepool_max = nn.AdaptiveMaxPool1d(dim_emb_transformer)
        # self.adaptivepool_min = nn.AdaptiveMinPool2d(dim_emb_transformer)
        self.te = TransformerEncoderLayer(dim_emb_transformer, nhead, dim_feedforward, dropout, activation)
        self.mlp = MLP(dim_emb_transformer, dim_feedforward=dim_emb_transformer**2, number_classes=number_classes)
        self.activation = nn.GELU()
        
    def forward(self, data):
        x = self.depthwiseconv(data.permute(1,2,0)).squeeze(0)#.permute(1,0)
        x = self.adaptivepool_max(x)
        x = self.te(x)
        print(x.shape)
        x = self.mlp(x)#[:,0])
        print(x.shape)
        x = self.activation(x)
        return x
    
def my_loss(out, label, batch):
    # loss 92 categories
    # return (torch.sum((torch.tensor(out.argmax(1).to(torch.float32), requires_grad = True)-label)**2)/batch)
    # loss 4 categories
    return torch.tensor(((abs(out - label)).sum().item())**2/(batch), requires_grad = True)


# nn.CrossEntropyLoss()
def training(net, train, optimizer, batch_size, criterion = my_loss, epochs: int = 10):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # set the network in the training mode
    net.train()

    for i in range(epochs):
        for data, label in tqdm(train):
            data , label = data.to(device), label.to(device)
            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs, label, batch_size) # criterion(outputs, label)
            loss.backward()
            optimizer.step()
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # calculate the loss and accuracy
            samples += len(data)
            cumulative_loss += loss.item()
            cumulative_accuracy += (outputs.int() == label).sum().item()/4
        
        # print statistics
        print('epoch: %d, loss: %.3f, accuracy: %.3f' % (i + 1, cumulative_loss / samples, cumulative_accuracy / samples))
        # print(f'loss: {round(cumulative_loss / samples,4)}')
        
def test(net, test, batch_size, criterion = my_loss, epochs: int = 10):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # set the network in the training mode
    net.eval()

    # disable gradient calculation
    with torch.no_grad():
        for data, label in tqdm(test):
            data, label = data.to(device), label.to(device)
            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs, label, batch_size)
        
            # calculate the loss and accuracy
            samples += len(data)
            cumulative_loss += loss.item()
            cumulative_accuracy += (outputs.int() == label).sum().item()/4
        
        # print statistics
        print('loss: %.3f, accuracy: %.3f' % (cumulative_loss / samples, cumulative_accuracy / samples))
        # print(f'loss: {round(cumulative_loss / samples,4)}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)     


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

