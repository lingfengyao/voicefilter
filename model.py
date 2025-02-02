import torch
import torch.nn as nn
import torch.nn.functional as F

n_fft = 1024
num_freq = n_fft // 2 + 1
fc2_dim = num_freq
emb_dim = 256
lstm_dim = 400
fc1_dim = 600


class VoiceFilter(nn.Module):
    def __init__(self):
        super(VoiceFilter, self).__init__()         

        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)), # (9, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)), # (17, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn6
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)), # (33, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn7
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)), # (65, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)), 
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            8*num_freq + emb_dim,
            lstm_dim,
            batch_first=True,
            bidirectional=True)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(2*lstm_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv(x)
        # x: [B, 8, T, num_freq]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        # dvec: [B, emb_dim]
        dvec = dvec.unsqueeze(1)
        dvec = dvec.repeat(1, x.size(1), 1)
        # dvec: [B, T, emb_dim]

        x = torch.cat((x, dvec), dim=2) # [B, T, 8*num_freq + emb_dim]        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.dropout(x)                   
        x = self.fc1(x) # x: [B, T, fc1_dim]
    
        x = F.relu(x)
        x = self.fc2(x)  # [B, T, fc2_dim]
        x = torch.sigmoid(x)
        return x

