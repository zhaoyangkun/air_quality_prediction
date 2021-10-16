import pandas as pd
import torch
import numpy as np
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pickle

# 原始数据预处理
ori = pd.read_excel(
    'LSTM/data/beijing.xlsx', header=None)

data = ori.drop(columns=2)

col_names = ['date', 'aqi', 'pm2', 'pm10', 'so2', 'co', 'no2', 'o3']

data.columns = ['date', 'aqi', 'pm2', 'pm10', 'so2', 'co', 'no2', 'o3']

pro = data.iloc[:, 1:]

for name in col_names[1:]:
    pro[name] = (pro[name] - pro[name].min()) / \
                (pro[name].max() - pro[name].min())  # 归一化处理

pro['date'] = data['date']

pro.to_excel('LSTM/data/pro_data.xlsx', index=False)

# pro = pd.read_excel('LSTM/data/pro_data.xlsx')

# 设置超参数
x_timesteps = 5  # 用多少期去预测（在RNN模型中x_timesteps就是cell的个数）
y_timesteps = 1  # 老师说一般都只预测一期，所以y_timestpes应该就是固定值1，但是后面Env我懒得改了，所以这里还是保留了y_timesteps这个超参数
stride = 2  # 每次移动多少期来采样
hidden_size = 20
hidden_layers = 1
y_features = 7  # 最终你想要预测多少个特征，比如我用前5期的7个特征预测滞后期的7个特征，那么y_features就是7；如果只想预测其中某个特征，那么y_features就是1
if y_features < 7:
    the_col_wanted = [int(x) for x in input(
        '输入您想要的列，若超过1列，请用,隔开（英文逗号）。您选择的列是：').split(',')]  # 你想要预测的特征，这个长度必须与y_features一致
    if len(the_col_wanted) == y_features:
        print('您最终选择的列是：', the_col_wanted)
    else:
        print('您的选择有误,请重新进行选择')
else:
    the_col_wanted = list(range(7))

batch_size = 32
epochs = 200


class Net(nn.Module):
    """搭建网络"""

    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size,
                            num_layers=hidden_layers, batch_first=True)
        # 注意这里指定了batch_first为true哈
        # 这里设置了两个线性层，其实设置一层也可以
        self.linear1 = nn.Linear(
            in_features=hidden_size, out_features=int(hidden_size / 2))
        self.linear2 = nn.Linear(in_features=int(
            hidden_size / 2), out_features=y_features)

    def forward(self, x, h0):
        out, (h, c) = self.lstm(x)
        # x 的 size 是 batch_size*x_timesteps*x_features   本例中的 x_features 是 7
        # LSTM 的最终的输出是 3 个，h 和 c 都是最后一个时刻的 h、c
        # out 的 size 是 batch_size*x_timesteps*hidden_size
        # h 和 c 的 size 是 (num_directions*num_layers, batch_size, hidden_size)。
        # 注意，不管有没有设置 batch_first=True, batch_size 永远在 h 和 c 的 size 的第二个。
        # （而设置了 batch_first=True 之后，batch_size 在 output 的 size 的第一个）
        # 只要最后一个 cell 的输出结果，out 的 size 变为 batch_size*1*hidden_size
        out = out[:, -1, :]
        out = out.reshape(batch_size, -1)
        out = self.linear1(out)
        out = self.linear2(out).reshape(batch_size, 1, y_features)
        # 经过整个网络之后，size 由 batch_size*x_timesteps*x_features变成了 batch_size*1*y_features
        return out


class Env(Dataset):
    """
    创建数据集
    """

    def __init__(self, root, x_timesteps, y_timesteps, stride, mode):
        super(Env, self).__init__()
        self.data = pd.read_excel(root).iloc[:, :-1].values
        self.x_timesteps = x_timesteps
        self.y_timesteps = y_timesteps
        self.stride = stride
        self.mode = mode
        self.samples = self.creat_xy('./final_sample.pkl')

        self.x = self.samples[:, :-self.y_timesteps, :]
        if self.y_timesteps == 1:
            self.y = self.samples[:, -1,
                     the_col_wanted].reshape(len(self.x), 1, y_features)
        else:
            self.y = self.samples[:, -self.y_timesteps:, the_col_wanted]

        if self.mode == 'train':
            self.x = self.x[:int(0.6 * len(self.x)), :, :]
            self.y = self.y[:int(0.6 * len(self.y)), :, :]

        if self.mode == 'val':
            self.x = self.x[int(0.6 * len(self.x)):int(0.8 * len(self.x)), :, :]
            self.y = self.y[int(0.6 * len(self.y)):int(0.8 * len(self.y)), :, :]

        if self.mode == 'test':
            self.x = self.x[int(0.8 * len(self.x)):, :, :]
            self.y = self.y[int(0.8 * len(self.y)):, :, :]

    def creat_xy(self, save_path):
        # 此函数用于创造sample，每个样本的size是x_timesteps+y_timesteps*7
        # 前面的x_timesteps*7就是放入网络中的每个样本，后面的y_timestps*7就是原始的true_y
        index = 0
        samples = []

        while (index + self.x_timesteps + self.y_timesteps) <= (len(self.data) - 1):
            single_sample = self.data[index: index +
                                             self.x_timesteps + self.y_timesteps, :]
            samples.append(single_sample)
            # 每个single_sample的size是x_timesteps+y_timesteps*7
            # 前面的x_timesteps*7就是放入网络中的每个样本，后面的y_timestps*7就是原始的true_y
            index += self.stride
        else:
            final_sample = torch.from_numpy(np.array(samples))
            with open(save_path, 'wb') as f:  # 将数据写入pkl文件
                pickle.dump(final_sample, f)
            return final_sample

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx, :, :], self.y[idx, :, :]

        return x, y


# 创建数据集


class Env(Dataset):
    def __init__(self, root, x_timesteps, y_timesteps, stride, mode):
        super(Env, self).__init__()
        self.data = pd.read_excel(root).iloc[:, :-1].values
        self.x_timesteps = x_timesteps
        self.y_timesteps = y_timesteps
        self.stride = stride
        self.mode = mode
        self.samples = self.creat_xy('./final_sample.pkl')

        self.x = self.samples[:, :-self.y_timesteps, :]
        if self.y_timesteps == 1:
            self.y = self.samples[:, -1,
                     the_col_wanted].reshape(len(self.x), 1, y_features)
        else:
            self.y = self.samples[:, -self.y_timesteps:, the_col_wanted]

        if self.mode == 'train':
            self.x = self.x[:int(0.6 * len(self.x)), :, :]
            self.y = self.y[:int(0.6 * len(self.y)), :, :]

        if self.mode == 'val':
            self.x = self.x[int(0.6 * len(self.x)):int(0.8 * len(self.x)), :, :]
            self.y = self.y[int(0.6 * len(self.y)):int(0.8 * len(self.y)), :, :]

        if self.mode == 'test':
            self.x = self.x[int(0.8 * len(self.x)):, :, :]
            self.y = self.y[int(0.8 * len(self.y)):, :, :]

    def creat_xy(self, save_path):
        # 此函数用于创造sample，每个样本的size是x_timesteps+y_timesteps*7
        # 前面的x_timesteps*7就是放入网络中的每个样本，后面的y_timestps*7就是原始的true_y
        index = 0
        samples = []

        while (index + self.x_timesteps + self.y_timesteps) <= (len(self.data) - 1):
            single_sample = self.data[index: index +
                                             self.x_timesteps + self.y_timesteps, :]
            samples.append(single_sample)
            # 每个single_sample的size是x_timesteps+y_timesteps*7
            # 前面的x_timesteps*7就是放入网络中的每个样本，后面的y_timestps*7就是原始的true_y
            index += self.stride
        else:
            final_sample = torch.from_numpy(np.array(samples))
            with open(save_path, 'wb') as f:  # 将数据写入pkl文件
                pickle.dump(final_sample, f)
            return final_sample

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx, :, :], self.y[idx, :, :]

        return x, y


# 准备好数据
train_db = Env('LSTM/data/pro_data.xlsx', x_timesteps,
               y_timesteps, stride, 'train')
val_db = Env('LSTM/data/pro_data.xlsx', x_timesteps,
             y_timesteps, stride, 'val')
test_db = Env('LSTM/data/pro_data.xlsx', x_timesteps,
              y_timesteps, stride, 'test')
train_loader = DataLoader(train_db, batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_db, batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_db, batch_size, shuffle=False, drop_last=True)

# 初始化模型、定义损失函数、优化器
model = Net()
h0, c0 = torch.zeros([hidden_layers, batch_size, hidden_size]), torch.zeros(
    [hidden_layers, batch_size, hidden_size])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
best_loss = 99999  # 因为希望val_loss不断减小，所以初始的val_loss设置大一点


# 设置一个evaluate函数，用于评估模型的效果(这里使用loss来衡量，根据实际情况，也可以选择precision、recall、F_β score、auc等来评估)
def evaluate(loader_name):
    loss_for_all_batch = []
    for batch_index, (x, y) in enumerate(loader_name):
        input_x = x.float()
        true_y = y.float()
        with torch.no_grad():
            pre_y = model.forward(input_x, (h0, c0))
            loss = loss_fn(pre_y, true_y)  # 每个batch的loss
            loss_for_all_batch.append(loss)
    # 用所有batch loss的均值代表该数据集上的总体loss水平
    loss_for_this_loader = np.mean(loss_for_all_batch)
    return loss_for_this_loader


# 开始训练
for it in range(epochs):
    for batch_index, (x, y) in enumerate(train_loader):
        input_x = x.float()
        true_y = y.float()
        pre_y = model.forward(input_x, (h0, c0))
        loss = loss_fn(pre_y, true_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_index + 1) % 10 == 0:
            print('epoch：', it + 1, '   batch_index:',
                  batch_index + 1, '  loss:', loss.item())

    # 每隔两个epoch就在val上看一下效果
    if (it + 1) % 2 == 0:
        loss_for_val = evaluate(val_loader)
        if loss_for_val < best_loss:
            print('已经完 成了{}次迭代，val的loss有所下降,val_loss为：{}'.format(it + 1, loss_for_val))
            best_epoch = it + 1
            best_loss = loss_for_val
            torch.save(model.state_dict(), 'best_model_ckp.txt')

print('模型已训练完成，最好的epoch是{}，在验证集上的loss是{}'.format(best_epoch, best_loss))

model.load_state_dict(torch.load('best_model_ckp.txt'))
print('已将参数设置成训练过程中的最优值，现在开始测试test_set')
loss_for_test = evaluate(test_loader)
print('测试集上的loss为：', loss_for_test)
