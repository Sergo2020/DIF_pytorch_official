import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image



def check_existence(path, create=False):
    if not os.path.exists(path):
        print("Creating check point directory - " + str(path))

        if create:
            os.mkdir(path)
        else:
            print(f'{str(path)}\nPath not found')
            exit()


def show_image(img, counters=None, centers=None, rectangles=None, title='Image',
               path=None,
               colors='gray'):  # Simple function that shows image in pre set image size without axis and grid
    plt.figure(figsize=(6, 6), frameon=False)
    img_t = img.copy()

    if centers is not None:
        for idx in range(1, len(centers) + 1):
            cv.putText(img_t, str(idx), tuple(centers[idx - 1]), cv.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 3,
                       cv.LINE_AA)

    if rectangles is not None:
        for r in rectangles:
            cv.rectangle(img_t, r[0], r[1], color=(255, 0, 0), thickness=3)

    if counters is not None:
        for cnt in counters:
            cv.drawContours(img_t, cnt, -1, (255, 0, 0), thickness=3)

    if len(img.shape) < 3:
        plt.imshow(img_t, colors)
    else:
        plt.imshow(img_t)
    plt.grid(False)
    plt.axis(False)
    if path:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    else:
        plt.title(title)
        plt.show()


def calc_even_size(img_size, d):
    d = int(np.power(2, d))
    h, w = img_size

    if h % 2 != 0:
        h -= 1
    if w % 2 != 0:
        w -= 1

    d_h = (h % d) // 2
    d_w = (w % d) // 2

    return d_h, h - d_h, d_w, w - d_w


def make_even(img, d):  # Force image size to power of 2
    d_h, n_h, d_w, n_w = calc_even_size(img.shape[2:], d)
    return img[:, :, d_h:n_h, d_w:n_w]


def produce_spectrum(img_np):
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_fft = np.fft.fft2(img_np, axes=(0, 1), norm='forward')
    magnitude_spectrum = 20 * np.log(np.abs(np.fft.fftshift(img_fft)).mean(2) + 1e-8)

    return magnitude_spectrum
