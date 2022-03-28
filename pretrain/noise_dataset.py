from io import BytesIO
import numpy as np
import cv2

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def renormalize(x, dtype='int'):
    if dtype == 'int':
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = 255 * x
        x = x.astype(np.uint8)
    else:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return x


def generate_pink_noise(size=(256, 256), alpha=0.5, beta=3.5, eps=1e-6):
    # generate white noise
    normal_noise = np.random.randn(*size)

    # fft
    normal_noise_fft = np.fft.fft2(normal_noise)

    freq = np.abs(np.fft.fftfreq(normal_noise_fft.shape[0], d=0.2))
    # remove zero (division by zero)
    freq[0] = freq[1] * 0.1

    # exponent in [0.5, 3.5]
    a, b = (np.random.rand(2)) * (beta - alpha) + alpha

    freq_x = np.float_power(freq, a) + eps
    freq_y = np.float_power(freq, b) + eps

    freq_x = np.expand_dims(freq_x, 1)
    freq_y = np.expand_dims(freq_y, 0)

    freq_x = np.repeat(freq_x, size[0], 1)
    freq_y = np.repeat(freq_y, size[1], 0)

    w = 1 / (freq_x + freq_y)

    pink_noise_fft = w * normal_noise_fft

    pink_noise = np.fft.ifft2(pink_noise_fft).real

    return pink_noise, normal_noise


def generate_rgb_noise(size=(256, 256), alpha=0.5, beta=3.5, eps=1e-6):
    color_noise_map = []
    color_noise_norm = []
    for _ in range(3):
        tmp_pink, tmp_norm = generate_pink_noise(size, alpha, beta, eps)
        color_noise_map.append(tmp_pink)
        color_noise_norm.append(tmp_norm)
    color_noise_map = np.stack(color_noise_map, 2)
    color_noise_norm = np.stack(color_noise_norm, 2)
    return color_noise_map, color_noise_norm


def inject_saliency(size, canvas):
    shape_list = ['line', 'circle', 'circle', 'ellipse', 'ellipse', 'rectangle', 'rectangle']
    last_shape = shape_list[np.random.randint(1, len(shape_list))]
    b, g, r = np.random.randint(0, 255, size=(3,), dtype=np.uint8).tolist()
    x, y = np.random.randint(size[0]//2 - size[0]//10, size[0]//2 + size[0]//10, size=(2,), dtype=np.uint8).tolist()
    if last_shape == 'circle':
        rad = np.random.randint(size[0]//4, 3*size[0]//8, dtype=np.uint8).tolist()
        canvas = cv2.circle(canvas, (x, y), rad, (b, g, r), -1)
    elif last_shape == 'ellipse':
        r1, r2 = np.random.randint(size[0]//4, 3*size[0]//8, size=(2,), dtype=np.uint8).tolist()
        canvas = cv2.ellipse(canvas, (x, y), (r1, r2), 0, 0, 360, (b, g, r), -1)
    else:
        x1, y1 = np.random.randint(size[0]//8, size[0]//2, size=(2,), dtype=np.uint8).tolist()
        h, w = np.random.randint(3*size[0]//8, 3*size[0]//4, size=(2,), dtype=np.uint8).tolist()
        x2, y2 = (x1 + h, y1 + w)
        canvas = cv2.rectangle(canvas, (x1, y1), (x2, y2), (b, g, r), -1)
    return canvas


def inject_particles(size, canvas, cur=0, iterations=100):
    shape_list = ['line', 'circle', 'circle', 'ellipse', 'ellipse', 'rectangle', 'rectangle']
    shape = shape_list[np.random.randint(len(shape_list))]

    # color
    b, g, r = np.random.randint(0, 255, size=(3,), dtype=np.uint8).tolist()
    max_size = int(np.round(size[0] // 5 * (iterations - cur) / iterations + 2))
    if shape == 'line':
        # two points
        x1, x2, y1, y2 = np.random.randint(0, 255, size=(4,), dtype=np.uint8).tolist()
        x_tl, x_br = (x1, x2) if x1 < x2 else (x2, x1)
        y_tl, y_br = (y1, y2) if y1 < y2 else (y2, y1)
        # thickness
        t = np.random.randint(1, max_size)
        # draw
        canvas = cv2.line(canvas, (x_tl, y_tl), (x_br, y_br), (b, g, r), t)
    elif shape == 'circle':
        # center and radius
        x, y = np.random.randint(low=0, high=size[0], size=(2,), dtype=np.uint8).tolist()
        rad = np.random.randint(low=1, high=max_size, dtype=np.uint8).tolist()
        canvas = cv2.circle(canvas, (x, y), rad, (b, g, r), -1)
    elif shape == 'ellipse':
        # center and radius
        x, y = np.random.randint(low=0, high=size[0], size=(2,), dtype=np.uint8).tolist()
        r1, r2 = np.random.randint(low=1, high=max_size, size=(2,), dtype=np.uint8).tolist()
        ang = np.random.randint(0, 180)
        canvas = cv2.ellipse(canvas, (x, y), (r1, r2), ang, 0, 360, (b, g, r), -1)
    elif shape == 'rectangle':
        x1, y1 = np.random.randint(0, 255 - max_size, size=(2,), dtype=np.uint8).tolist()
        h, w = np.random.randint(1, max_size, size=(2,), dtype=np.uint8).tolist()
        x2, y2 = (x1 + h, y1 + w)
        canvas = cv2.rectangle(canvas, (x1, y1), (x2, y2), (b, g, r), -1)
    else:
        print("{} is not supported")
        exit()
    return canvas


def pink_leaves(size=(256, 256, 3), iterations=100):
    canvas, _ = generate_rgb_noise(size[:2])
    canvas = renormalize(canvas)
    canvas = np.ascontiguousarray(canvas)
    for i in range(iterations):
        canvas = inject_particles(size, canvas, i, iterations)
        # print(canvas.dtype)
        # injection = np.zeros(size)
        # injection = inject_particles(size, injection, i, iterations)
        # mask = injection.copy()
        # mask[mask > 0] = 1
        # inside_pattern, _ = generate_rgb_noise(size[:2])
        # inside_pattern = renormalize(inside_pattern)
        # canvas = (1 - mask) * canvas + mask * inside_pattern
        # print(np.max(canvas), np.min(canvas))

    injection = np.zeros(size)
    injection = inject_saliency(size, injection)
    mask = injection.copy()
    mask[mask > 0] = 1
    inside_pattern, _ = generate_rgb_noise(size[:2])
    inside_pattern = renormalize(inside_pattern)
    canvas = (1 - mask) * canvas + mask * inside_pattern

    return canvas


def pink_and_salient(size=(256, 256, 3), pink_inside=True):
    canvas, _ = generate_rgb_noise(size[:2])
    canvas = renormalize(canvas)
    if pink_inside:
        saliency = np.zeros(size)
        saliency = inject_saliency(size, saliency)
        mask = saliency.copy()
        mask[mask > 0] = 1
        inside_pattern, _ = generate_rgb_noise(size[:2])
        inside_pattern = renormalize(inside_pattern)
        canvas = (1 - mask) * canvas + mask * inside_pattern
    else:
        canvas = inject_saliency(size, canvas)
    return canvas


def dead_leaves(size=(256, 256, 3), iterations=100):
    back_colors = np.random.randint(0, 255, size=(3,), dtype=np.uint8).tolist()
    canvas = np.zeros(size)
    for i in range(len(back_colors)):
        canvas[:, :, i:i+1] += back_colors[0]

    for i in range(iterations):
        canvas = inject_particles(size, canvas, i, iterations)

    canvas = inject_saliency(size, canvas)
    return canvas


def get_fft_mag_phase(img):
    ch_fft = np.fft.fft2(img)
    mag = np.abs(ch_fft)
    phase = np.angle(ch_fft)
    return mag, phase


def inverse_fft(mag, phase):
    out = np.multiply(mag, np.exp(1j*phase))
    return np.fft.ifft2(out).real


def swap_mag_phase_rgb(mag_src, ph_src):
    res = []
    for i in range(3):
        mag, _ = get_fft_mag_phase(mag_src[:, :, i])
        _, ph = get_fft_mag_phase(ph_src[:, :, i])
        tmp = inverse_fft(mag, ph)
        res.append(tmp)
    res = np.stack(res, 2)
    return res


class NoiseDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.length = 500
        self._name = 'PinkNoise'
        self._raw_shape = [500, 3, 256, 256]
        self._use_labels = False
        self._raw_labels = None
        self._label_shape = None
        max_size = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # img, _ = generate_rgb_noise()
        # dead_leaf = dead_leaves(iterations=100)
        # img = swap_mag_phase_rgb(img, dead_leaf)

        img = pink_leaves()
        # img = pink_and_salient()

        img = torch.from_numpy(img)
        img = img.permute((2, 0, 1))
        img = img.float()

        return img, self.get_label(index)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return [3, 256, 256]

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64
