import os
import pickle
import random

import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
from torch.utils.data import Dataset, DataLoader


class SSLR_Dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.total_x, self.total_y, self.total_z = self._read_file()

    def __getitem__(self, index):
        return torch.tensor(self.total_x[index], dtype=torch.float), torch.tensor(self.total_y[index], dtype=torch.float), torch.tensor(self.total_z[index], dtype=torch.int)

    def __len__(self):
        return len(self.total_x)

    def encode(self, y):
        def gaussian_func(gt_angle):
            # sigma = beam_width
            sigma = 3.15
            # sigma = 8
            angles = np.arange(360)
            out = np.array(np.exp(-1 * np.square(angles - 180) / sigma ** 2))
            out = np.roll(out, gt_angle - 180) # 向右 roll gt_angle - 180 / 向左 roll 180 - gt_angle
            return out

        mat_out = []
        for gt_angle in y:
            if not np.isnan(gt_angle):
                mat_out.append(gaussian_func(gt_angle))
        if not mat_out:
            return np.full(360, 0)
        # mat_out对360个角度分别取max
        mat_out = np.asarray(mat_out)
        mat_out = mat_out.transpose()
        mat_out = np.max(mat_out, axis=1)
        return mat_out

    def _read_file(self):
        frames = os.listdir(self.data_path)
        total_x = []
        total_y = []
        total_z = [] # num of sources
        print("read_file starts")
        for frame in frames:
            try:
                with open(os.path.join(self.data_path, frame), 'rb') as file:
                    origin_data = pickle.load(file)
                z = origin_data["num_sources"]
                if z == 0:
                    continue
                total_z.append(z)  # [len(frames)]

                y = []
                if origin_data["label_seg_level"][0] == 360:
                    origin_data["label_seg_level"][0] = 0
                y.append(origin_data["label_seg_level"][0])
                if origin_data["label_seg_level"][1] == 360:
                    origin_data["label_seg_level"][1] = 0
                y.append(origin_data["label_seg_level"][1])
                likelihood_coding = self.encode(y) # [360]

                total_y.append(likelihood_coding)  # [len(frames), 360]

                x = origin_data["stft_seg_level"]
                total_x.append(x)  # [len(frames), 8, 7, 337]

                # z = origin_data["num_sources"]
                # total_z.append(z)  # [len(frames)]

                if len(total_z) % 10000 == 0:
                    print("{} frames have been processed".format(len(total_z)), flush=True)
            except Exception as e:
                print(f'Exception {e}')
        print(f'total_samples {len(total_z)}')

        return total_x, total_y, total_z

if __name__ == '__main__':
    train_data_path = "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/test_data_frame_level_single_source"
    train_data = DataLoader(SSLR_Dataset(train_data_path), batch_size=64, shuffle=True, num_workers=4) # train_data.shape (batch_x, batch_y)
    print(len(train_data)) # len(train_data) is samples / batch_size
    print(next(iter(train_data))[0].shape, next(iter(train_data))[1].shape)



