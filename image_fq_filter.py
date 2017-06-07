import numpy as np
import cv2
import sys

def dft(img):
    freqs = np.fft.fft2(img)
    return np.fft.fftshift(freqs)

def idft(fshift):
    freqs = np.fft.ifftshift(fshift)
    return np.fft.ifft2(freqs)

def magnitude(fshift):
    mag = np.log(np.abs(fshift))
    return normalize(mag)

def normalize(img):
    return img/img.max();

def edit_freq(x, y, is_fill):
    global src_freqs, valid_freqs

    n = 30
    y = max(0, y-n//2)
    x = max(0, x-n//2)
    if is_fill:
        valid_freqs[y:y+n, x:x+n] = src_freqs[y:y+n, x:x+n]
    else:
        valid_freqs[y:y+n, x:x+n] = 0

def show():
    global valid_freqs

    img_back = normalize(idft(valid_freqs).real)
    cv2.imshow('filtered', img_back)
    mag = magnitude(valid_freqs)
    cv2.imshow('mag', mag)

def onmouse(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing == True:
        edit_freq(x, y, flags&cv2.EVENT_FLAG_SHIFTKEY)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    show()

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img_float = normalize(np.float64(img))
    rows, cols = img.shape
    drawing = False

    src_freqs = dft(img_float)
    valid_freqs = dft(img_float)

    edit_freq(rows//2, cols//2, True)

    show()
    cv2.imshow('original', img_float)
    cv2.moveWindow('original', 0, 0)
    cv2.moveWindow('filtered', cols, 0)
    cv2.moveWindow('mag', cols*2, 0)
    cv2.setMouseCallback('mag',onmouse)

    cv2.waitKey(0)

