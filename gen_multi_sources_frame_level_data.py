import argparse
import math
import os
from pathlib import Path
import numpy as np
import pickle
import apkit

_FREQ_MAX = 8000
_FREQ_MIN = 100
SEG_LEN = 8192
SEG_HOP = 4096

def main(gt_seg_path, audio_dir_path, data_frame_path):
    '''
    gt_seg_path: "/CDShare2/SSLR/lsp_test_106_w8192/gt_frame"
    audio_dir_path: "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/lsp_test_106/audio"  # .wav音频所在目录
    data_frame_path = "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/test_data_frame_level" # 每帧的特征和标签
    '''

    if not os.path.exists(data_frame_path):
        os.makedirs(data_frame_path)

    gts = list(Path(gt_seg_path).rglob('*.txt'))
    cnt_segs = 0 # gt_frame actually means gt_segment 
    for gt in gts:
        gt = str(gt)
        audio_id = gt.split('/')[-1].split('.')[0].replace('qianspeech_', '')
        print('gt: ' + gt, flush=True)
        audio_path = os.path.join(audio_dir_path, audio_id + '.wav')
        print('audio: ' + audio_path, flush=True)
        # load signal
        fs, sig = apkit.load_wav(audio_path)  # sig.shape: [C, length] ndarray(float64)

        feat_seg_idx = 0
        with open(gt, "r") as f:
            lines = f.readlines()
            for line in lines:
                gt_seg_data = line.split(' ')
                feat_seg_idx = int(gt_seg_data[0])
                print(f'gt_seg_data {gt_seg_data}', flush=True)

                if gt_seg_data[8] == '1\n':
                    num_sources = 2
                else:
                    num_sources = 1
                # print(f'num_sources {num_sources}', flush=True)
                doa_gt_1 = np.arctan2(float(gt_seg_data[1]), float(gt_seg_data[2]))
                doa_gt_1 = round(math.degrees(doa_gt_1)) + 180

                doa_gt_2 = np.nan
                if num_sources == 2:
                    doa_gt_2 = np.arctan2(float(gt_seg_data[4]), float(gt_seg_data[5]))
                    doa_gt_2 = round(math.degrees(doa_gt_2)) + 180

                label_seg_level = [doa_gt_1, doa_gt_2]
                
                # calculate the complex spectrogram stft
                tf = apkit.stft(sig[:, feat_seg_idx * SEG_HOP : feat_seg_idx * SEG_HOP + SEG_LEN], apkit.cola_hamming, 2048, 1024, last_sample=True)
                # tf.shape: [C, num_frames, win_size] tf.dtype: complex128
                nch, nframe, _ = tf.shape # num_frames=sig_len/hop_len - 1
                # tf.shape:(4, num_frames, 2048) num_frames=7 when len_segment=8192 and win_size=2048
                # why not Nyquist 1 + n_fft/ 2?

                # trim freq bins
                max_fbin = int(_FREQ_MAX * 2048 / fs)            # 100-8kHz
                min_fbin = int(_FREQ_MIN * 2048 / fs)            # 100-8kHz
                tf = tf[:, :, min_fbin:max_fbin]
                # tf.shape: (C, num_frames, 337)

                # calculate the real part of the spectrogram
                real_spectrogram = tf.real
                # real_spectrogram.shape: (C, num_frames, 337) real_spectrogram.dtype: float64

                # calculate the imaginary part of the spectrogram
                imaginary_spectrogram = tf.imag
                # imaginary_spectrogram.shape: (C, num_frames, 337) imaginary_spectrogram.dtype: float64

                # combine these two parts by the channel axis
                stft_seg_level = np.concatenate((real_spectrogram, imaginary_spectrogram), axis=0)
                # stft_seg_level.shape: (C*2, num_frames, 337) stft_seg_level.dtype: float64

                # sample_data 同时有特征和标签
                sample_data = {"stft_seg_level" : stft_seg_level, "label_seg_level" : label_seg_level, "num_sources" : num_sources}
                save_path = os.path.join(data_frame_path, '{}_seg_{}.pkl'.format(audio_id, feat_seg_idx))
                # print(save_path, flush=True)
                print("sample_data's feat.shape {}".format(sample_data["stft_seg_level"].shape), flush=True)
                print("sample_data's label {}".format(sample_data["label_seg_level"]), flush=True)
                pkl_file = open(save_path, 'wb')
                pickle.dump(sample_data, pkl_file)
                pkl_file.close()
                cnt_segs += 1

    print("cnt_segs: {}".format(cnt_segs), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate multi sources data at frame level')
    parser.add_argument('gt_frame', metavar='GT_FRAME_PATH', type=str,
                        help='path to the gt_frame directory')
    parser.add_argument('audio_dir', metavar='AUDIO_DIR_PATH', type=str,
                        help='path to the audio_dir directory')
    parser.add_argument('data_frame', metavar='DATA_FRAME_PATH', type=str,
                        help='path to the data_frame directory')
    args = parser.parse_args()
    main(args.gt_frame, args.audio_dir, args.data_frame)
