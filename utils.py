# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

ref_level_db = 20
min_level_db = -100

def parameters():
    n_fft = 1024
    hop_length = 513
    win_length = 1024
    return n_fft, hop_length, win_length

def stft(y):
    n_fft, hop_length, win_length = parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def istft(mag, phase):
    stft_matrix = mag * np.exp(1j*phase)
    n_fft, hop_length, win_length = parameters()
    return librosa.istft(stft_matrix, hop_length=hop_length, win_length=win_length)

def wav2spec(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - ref_level_db
    S, D = normalize(S), np.angle(D)
    S, D = S.T, D.T 
    return S, D

def spec2wav(spectrogram, phase):
    spectrogram, phase = spectrogram.T, phase.T
    S = db_to_amp(denormalize(spectrogram) + ref_level_db)
    return istft(S, phase)

def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def normalize(S):
    return np.clip(S / -min_level_db, -1.0, 0.0) + 1.0

def denormalize(S):
    return (np.clip(S, 0.0, 1.0) - 1.0) * -min_level_db


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data

class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation(self, test_loss, sdr,
                       mixed_wav, target_wav, est_wav,
                       mixed_spec, target_spec, est_spec, est_mask,
                       epoch):
        
        self.add_scalar('test_loss', test_loss, epoch)
        self.add_scalar('SDR', sdr, epoch)

        self.add_audio('mixed_wav', mixed_wav, epoch, 16000)
        self.add_audio('target_wav', target_wav, epoch, 16000)
        self.add_audio('estimated_wav', est_wav, epoch, 16000)

        self.add_image('data/mixed_spectrogram',
            plot_spectrogram_to_numpy(mixed_spec), epoch, dataformats='HWC')
        self.add_image('data/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), epoch, dataformats='HWC')
        self.add_image('result/estimated_spectrogram',
            plot_spectrogram_to_numpy(est_spec), epoch, dataformats='HWC')
        self.add_image('result/estimated_mask',
            plot_spectrogram_to_numpy(est_mask), epoch, dataformats='HWC')
        self.add_image('result/estimation_error_sq',
            plot_spectrogram_to_numpy(np.square(est_spec - target_spec)), epoch, dataformats='HWC')