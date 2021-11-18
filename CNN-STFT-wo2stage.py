import os
import time
import numpy as np
import torch.nn as nn
import torch

from torch import optim

import prepare_multi_sources_data
import func
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader

train_data_path = "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/train_data_frame_level_stft"
test_data_path = "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/test_data_frame_level_stft"
model_save_path = "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/CNN-STFT"
device = torch.device('cuda:0')

# design model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # the input shape should be: (Channel=8, Time=7, Freq=337)
        # first conv and second conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, (1, 7), (1, 3), (0, 0)), # (32, 7, 110)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, (1, 5), (1, 2), (0, 0)), # (32, 7, 52)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128, affine=False), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128)
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(128, 360, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(360), nn.ReLU(inplace=True)
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(54, 500, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(500), nn.ReLU(inplace=True)
        )

        self.conv_8 = nn.Sequential(
            nn.Conv2d(500, 1, kernel_size=(7, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, a):
        # a.shape: [B, 8, 7, 337]
        a = self.conv1(a) # [B, 128, 7, 54]
        a_azi = self.relu(a+self.conv_1(a)) # [B, 128, 7, 54]
        a_azi = self.relu(a_azi+self.conv_2(a_azi)) # [B, 128, 7, 54]
        a_azi = self.relu(a_azi+self.conv_3(a_azi)) # [B, 128, 7, 54]
        a_azi = self.relu(a_azi+self.conv_4(a_azi)) # [B, 128, 7, 54]
        a_azi = self.relu(a_azi+self.conv_5(a_azi)) # [B, 128, 7, 54]
        a_azi0 = self.conv_6(a_azi) # [B, 360, 7, 54]
        a_azi = a_azi0.permute(0, 3, 2, 1) # [B, 54, 7, 360]
        a_azi = self.conv_7(a_azi) # [B, 500, 7, 360]
        a_azi = self.conv_8(a_azi) # [B, 1, 1, 360]
        a_azi = a_azi.view(a_azi.size(0), -1) # [B, 360]

        return a_azi0, a_azi

model = CNN()
model.to(device)

# construct loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_data = DataLoader(prepare_multi_sources_data.SSLR_Dataset(train_data_path), batch_size=100,
                        shuffle=True, num_workers=0)  # train_data is a tuple: (batch_x, batch_y)
test_data = DataLoader(prepare_multi_sources_data.SSLR_Dataset(test_data_path), batch_size=100,
                    shuffle=True, num_workers=0)  # test_data is a tuple: (batch_x, batch_y)

def main():
    ####################### Without Two-stage Training ##################################
    for epoch in range(40):
        # Train
        running_loss = 0.0

        # training cycle forward, backward, update
        iter = 0
        total_loss = 0.
        sam_size = 0.

        model.train()
        for (batch_x, batch_y, batch_z) in train_data:
            # 获得一个批次的数据和标签(inputs, labels)
            batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
            batch_y = batch_y.to(device) # batch_y.shape [B, 360]

            # 获得模型预测结果
            _, output = model(batch_x) # output.shape [B, 360]

            # 代价函数
            loss = criterion(output, batch_y) # averaged loss on batch_y

            running_loss += loss.item()
            if iter % 1000 == 0:
                now_loss = running_loss / 1000
                # scheduler.step(now_loss)
                print('[%d, %5d] loss: %.5f' % (epoch + 1, iter + 1, now_loss), flush=True)
                running_loss = 0.0
            with torch.no_grad():
                total_loss += loss.clone().detach().item() * batch_y.shape[0]
                sam_size += batch_y.shape[0]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 一个iter以一个batch为单位
            iter += 1
        
        scheduler.step()
        torch.cuda.empty_cache()
        save_path = os.path.join(model_save_path, 'Second_Stage_Epoch%d.pth'%(epoch+1))
        torch.save(model.state_dict(), save_path)
        # print the MSE and the sample size
        print(f'epoch {epoch + 1} loss {total_loss / sam_size} sam_size {sam_size}', flush=True)

        # Evaluate
        cnt_acc_single = 0
        cnt_acc_multi = 0
        sum_err_single = 0
        sum_err_multi = 0
        total = 0
        total_single = 0
        total_multi = 0

        with torch.no_grad():
            model.eval()
            for (batch_x, batch_y, batch_z) in test_data:
                batch_x = batch_x.to(device) # batch_x.shape [B, 8, 7, 337]
                # batch_y = batch_y.to(device) # batch_y.shape [B, 360]

                # batch_y.shape[0] = batch_size
                total += batch_z.size(0)

                # 获得模型预测结果
                _, output = model(batch_x) # output.shape [B, 360]

                for batch in range(batch_z.size(0)):
                    # test for known number of sources
                    num_sources = batch_z[batch]

                    if num_sources == 0:
                        total -= 1
                    
                    if num_sources == 1:
                        pred = torch.max(batch_y[batch], 0)[1].item()
                        label = torch.max(output[batch], 0)[1].item()
                        abs_err = func.angular_distance(pred, label)

                        if abs_err <= 5:
                            cnt_acc_single += 1
                        sum_err_single += abs_err
                        total_single += 1
                    if num_sources == 2:
                        pred = func.get_top2_doa(output[batch])
                        label = np.where(batch_y[batch].numpy() == 1)[0]
                        error = func.angular_distance(pred.reshape([2, 1]), label.reshape([1, 2]))
                        if error[0, 0]+error[1, 1] <= error[1, 0]+error[0, 1]:
                            abs_err = np.array([error[0, 0], error[1, 1]])
                        else:
                            abs_err = np.array([error[0, 1], error[1, 0]])
                        cnt_acc_multi += np.sum(abs_err <= 5)
                        sum_err_multi += abs_err.sum()
                        total += 1
                        total_multi += 2
            cnt_acc = cnt_acc_single + cnt_acc_multi
            sum_err = sum_err_single + sum_err_multi
        print(f'total_single {total_single} total_multi {total_multi} total {total}')
        print('Single-source accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_single / total_single), flush=True)
        print('Single-source MAE on test set: %.3f ' % (sum_err_single / total_single), flush=True)
        print('Two-sources accuracy on test set: %.2f %% ' % (100.0 * cnt_acc_multi / total_multi), flush=True)
        print('Two-sources MAE on test set: %.3f ' % (sum_err_multi / total_multi), flush=True)             
        print('Overall accuracy on test set: %.2f %% ' % (100.0 * cnt_acc / total), flush=True)
        print('Overall MAE on test set: %.3f ' % (sum_err / total), flush=True)


if __name__ == '__main__':
    main()
