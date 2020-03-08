import cv2
import numpy as np


def get_gaussian_kernel(r, sigma):
    ys, xs = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
    k = ys**2 + xs**2
    return np.exp(-(xs**2 + ys**2)/(2 * sigma**2)) / (2 * np.pi * sigma**2)


if __name__ == "__main__":
    # 1. 2d gaussian convolution
    img = cv2.imread('lena.png', 0).astype(np.float64)
    h, w = img.shape

    k = 3
    r = (k-1)//2
    sigma = 1 / (2*np.log(2))
    kernel_2d = get_gaussian_kernel(r, sigma)

    img_padded = np.pad(img, ((r,), (r,)), 'constant')
    img_conv = np.zeros_like(img)

    for y in range(r, h+r):
        for x in range(r, w+r):
            img_conv[y-r, x-r] = np.sum(img_padded[y-r:y+r+1, x-r:x+r+1] * kernel_2d)
    cv2.imwrite('lena_conv.png', img_conv.astype(np.uint8))

    # 2. 1d derivative convolution
    kernel_1d = np.array([-0.5, 0, 0.5])
    img_x_der = np.zeros_like(img)
    img_y_der = np.zeros_like(img)

    img_conv_padded = np.pad(img_conv, ((r,), (r,)), 'constant')
    img_conv_x_der = np.zeros_like(img)
    img_conv_y_der = np.zeros_like(img)

    for y in range(r, h+r):
        for x in range(r, w+r):
            img_x_der[y-r, x-r] = np.abs(np.sum(img_padded[y, x-r:x+r+1] * kernel_1d))
            img_y_der[y-r, x-r] = np.abs(np.sum(img_padded[y-r:y+r+1, x] * kernel_1d))
            img_conv_x_der[y-r, x-r] = np.abs(np.sum(img_conv_padded[y, x-r:x+r+1] * kernel_1d))
            img_conv_y_der[y-r, x-r] = np.abs(np.sum(img_conv_padded[y-r:y+r+1, x] * kernel_1d))

    cv2.imwrite('lena_x_der.png', img_x_der.astype(np.uint8))
    cv2.imwrite('lena_y_der.png', img_y_der.astype(np.uint8))

    img_grad = np.sqrt(img_x_der**2 + img_y_der**2)
    img_conv_grad = np.sqrt(img_conv_x_der**2 + img_conv_y_der**2)
    cv2.imwrite('lena_grad.png', img_grad.astype(np.uint8))
    cv2.imwrite('lena_conv_grad.png', img_conv_grad.astype(np.uint8))
