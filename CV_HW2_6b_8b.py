import time
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def conv_forward_im2col(input_data, weights, bias, stride=1, pad=0):
    filter_num, _, filter_h, filter_w = weights.shape
    N, _, H, W = input_data.shape
    out_h = int(1 + (H + 2 * pad - filter_h) / stride)
    out_w = int(1 + (W + 2 * pad - filter_w) / stride)

    col = im2col(input_data, filter_h, filter_w, stride, pad)
    col_W = weights.reshape(filter_num, -1).T

    out = np.dot(col, col_W) + bias
    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    return out


def conv_forward_direct(input_data, weights, bias, stride=1, pad=0):
    N, C, H, W = input_data.shape
    F, _, HH, WW = weights.shape
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))

    padded_input = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    for n in range(N):
        for f in range(F):
            for ho in range(H_out):
                for wo in range(W_out):
                    ho_start = ho * stride
                    wo_start = wo * stride
                    ho_end = ho_start + HH
                    wo_end = wo_start + WW

                    window = padded_input[n, :, ho_start:ho_end, wo_start:wo_end]

                    out[n, f, ho, wo] = np.sum(window * weights[f]) + bias[f]

    return out


# TEST
input_data = np.random.rand(1, 3, 500, 500)  # batch size of 1, 3 channels, 500x500 image
weights = np.random.rand(2, 3, 3, 3)  # 2 filters, 3 channels, 3x3 filter size
bias = np.random.rand(2)  # bias for each filter

start_time = time.perf_counter()
out_im2col = conv_forward_im2col(input_data, weights, bias)
im2col_time = time.perf_counter() - start_time

start_time = time.perf_counter()
out_direct = conv_forward_direct(input_data, weights, bias)
direct_time = time.perf_counter() - start_time

if np.allclose(out_im2col, out_direct):
    print("Результаты равны")
else:
    print("Результаты отличаются")

print(f"Время выполнения im2col: {im2col_time} секунд")
print(f"Время выполнения direct: {direct_time} секунд")
