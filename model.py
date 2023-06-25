import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy


class Linear_QNet(nn.Module):
    def __init__(self, IN, hidden, OUT):
        super().__init__()
        self.linear1 = nn.Linear(IN, hidden)
        self.linear2 = nn.Linear(hidden, OUT)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='randomtest.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.crit = nn.MSELoss()
        self.ep_q = []
        self.ep_loss = []

    def train_step(self, s, a, r, next_s, d):
        s = torch.tensor(numpy.array(s), dtype=torch.float)
        next_s = torch.tensor(numpy.array(next_s), dtype=torch.float)
        a = torch.tensor(numpy.array(a), dtype=torch.long)
        r = torch.tensor(numpy.array(r), dtype=torch.float)

        if len(s.shape) == 1:
            s = torch.unsqueeze(s, 0)
            next_s = torch.unsqueeze(next_s, 0)
            a = torch.unsqueeze(a, 0)
            r = torch.unsqueeze(r, 0)
            d = (d, )

        prediction = self.model(s)

        target = prediction.clone()
        for idx in range(len(d)):
            Q_new = r[idx]
            if not d[idx]:
                Q_new = r[idx] + self.gamma * torch.max(self.model(next_s[idx]))

            target[idx][torch.argmax(a[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.crit(target, prediction)
        self.ep_loss.append(loss.item())
        loss.backward()
        self.optimizer.step()