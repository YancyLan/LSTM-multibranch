import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision import transforms, datasets



def get_Data():

    data1=pd.read_csv('./data/textfeature2.csv')
    data21=pd.read_csv('./data/graphfeature.csv')
    data22 = pd.read_csv('./data/graph_features_new(1).csv')
    data3=pd.read_excel('./data/basicfeature.xlsx')
    label=pd.read_excel('./data/label.xlsx')
    data = pd.concat([data1, data21, data22, data3,label], axis=1)
    print(data.head())
    print(label.head())
    return data,label

def normalization(data,label):

    mm_x=MinMaxScaler() # 导入sklearn的预处理容器
    mm_y=MinMaxScaler()
    data=data.values    # 将pd的系列格式转换为np的数组格式
    label=label.values
    data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    label=mm_y.fit_transform(label)
    return data,label,mm_y

def split_windows(data,seq_length):

    x=[]
    y=[]
    for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _x=data[i:(i+seq_length),:]
        _y=data[i+seq_length,-1]
        x.append(_x)
        y.append(_y)
    x,y=np.array(x),np.array(y)
    print('x.shape,y.shape=\n',x.shape,y.shape)
    return x,y

def split_data(x,y,split_ratio):

    train_size=int(len(y)*split_ratio)
    test_size=len(y)-train_size

    x_data=Variable(torch.Tensor(np.array(x)))
    y_data=Variable(torch.Tensor(np.array(y)))

    x_train=Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train=Variable(torch.Tensor(np.array(y[0:train_size])))
    y_test=Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    x_test=Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
    .format(x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape))

    return x_data,y_data,x_train,y_train,x_test,y_test

def data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size):

    num_epochs=n_iters/(len(x_train)/batch_size) # n_iters代表一次迭代
    num_epochs=int(num_epochs)
    train_dataset=Data.TensorDataset(x_train,y_train)
    test_dataset=Data.TensorDataset(x_test,y_test)
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=True) # 加载数据集,使数据集可迭代
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

    return train_loader,test_loader,num_epochs


class CNNmultiLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, batch_size, seq_length) -> None:
        super(CNNmultiLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=64, seq_len=3, input_size=3) ---> permute(0, 2, 1)
        # (64, 3, 3)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=2),
            # shape(7,--)  ->(64,3,2)
            nn.ReLU())
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_per1 = nn.LSTM(input_size=11, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_per2 = nn.LSTM(input_size=15, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_per3 = nn.LSTM(input_size=13, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = x[:,:,0:11].to('cuda:0')
        x2 = x[:, :, 11:26].to('cuda:0')
        x3 = x[:, :, 26:39].to('cuda:0')
        x = x.to('cuda:0')
        h_1 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to('cuda:0')
        c_1 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to('cuda:0')
        output_x1, _ = self.lstm_per1(x1, (h_1, c_1))
        output_x2, _ = self.lstm_per2(x2, (h_1, c_1))
        output_x3, _ = self.lstm_per3(x3, (h_1, c_1))

        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        batch_size, seq_len = x.size()[0], x.size()[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to('cuda:0')
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to('cuda:0')
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0))
        output = torch.cat((output,output_x1,output_x2,output_x3),dim=1)
        pred = self.fc(output)
        pred = pred[:, -1, :]
        return pred



# 参数设置
seq_length=5 # 时间步长
input_size=39
out=64
num_layers=6
hidden_size=12
batch_size=64
n_iters=5000
lr=0.001
output_size=1
split_ratio=0.9
moudle=CNNmultiLSTM(input_size,out,hidden_size,num_layers,output_size,batch_size,seq_length).to('cuda:0')
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(moudle.parameters(),lr=lr)
print(moudle)

data,label=get_Data()
data,label,mm_y=normalization(data,label)
x,y=split_windows(data,seq_length)
x_data,y_data,x_train,y_train,x_test,y_test=split_data(x,y,split_ratio)
train_loader,test_loader,num_epochs=data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size)
# train
iter=0
for epochs in range(num_epochs):
  for i,(batch_x, batch_y) in enumerate (train_loader):
    outputs = moudle(batch_x)
    optimizer.zero_grad()   # 将每次传播时的梯度累积清除
    # print(outputs.shape, batch_y.shape)
    loss = criterion(outputs.to('cpu'),batch_y) # 计算损失
    loss.backward() # 反向传播
    optimizer.step()
    iter+=1
    if iter % 100 == 0:
      print("iter: %d, loss: %1.5f" % (iter, loss.item()))


moudle.eval()
train_predict = moudle(x_data)



def result(x_data, y_data):
  moudle.eval()
  train_predict = moudle(x_data)

  data_predict = train_predict.cpu().data.numpy()
  y_data_plot = y_data.data.numpy()
  y_data_plot = np.reshape(y_data_plot, (-1,1))
  data_predict = mm_y.inverse_transform(data_predict)
  y_data_plot = mm_y.inverse_transform(y_data_plot)


  plt.plot(y_data_plot)
  plt.plot(data_predict)
  plt.legend(('real', 'predict'),fontsize='15')
  my_y_ticks = np.arange(0.04, 0.055, 0.001)
  plt.yticks(my_y_ticks)
  plt.show()

  print('MAE/RMSE')
  print(mean_absolute_error(y_data_plot, data_predict))
  print(np.sqrt(mean_squared_error(y_data_plot, data_predict) ))

result(x_data, y_data)
result(x_test,y_test)


