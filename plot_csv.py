import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv_path', type=str, default='6_0001.ascii.csv',help='CSV 文件地址')
parser.add_argument('--output_path', type=str, default='result.png',help='图片保存地址')
parser.add_argument('--win_len', type=int, default=100,help='窗口宽度，预估的圆形像素直径')
parser.add_argument('--th', type=int, default=400,help='截断值，低于该值的像素点将被忽略，便于计算重心，找到光斑的位置')

args = parser.parse_args()



def gaussian(x, mu, sigma, a):
    # gaussian Fit
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def get_small_window(array_data,raw_array_data):
    # get array data center and h_line_intensity, v_line_intensity
    x = np.arange(0, array_data.shape[1])
    y = np.arange(0, array_data.shape[0])
    x, y = np.meshgrid(x, y)
    x_center = np.sum(x * array_data) / np.sum(array_data)
    y_center = np.sum(y * array_data) / np.sum(array_data)
    # get small window
    data = raw_array_data[int(y_center-LEN):int(y_center+LEN), int(x_center-LEN):int(x_center+LEN)]
    
    # 计算中心点位置
    center_x = data.shape[0] // 2
    center_y = data.shape[1] // 2

    # 计算沿水平线和垂直线的强度分布
    h_line_intensity = data[center_x, :]
    v_line_intensity = data[:, center_y]

    return data,h_line_intensity,v_line_intensity,center_x,center_y

if __name__ == "__main__":
    LEN = args.win_len/2
    array_data = pd.read_csv(args.csv_path, header=None).values.astype(np.float32)[:,:-1]
    array_data_copy = array_data.copy()
    array_data_copy[array_data_copy<args.th]=0
    data, h_line_intensity, v_line_intensity,center_x,center_y = get_small_window(array_data_copy,array_data)
    # 创建画布
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # 绘制矩阵
    ax1.set_title('Matrix')
    ax1.imshow(data, cmap='rainbow')
    # 将ax1的大小设置为其他三个子图的大小
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    ax1.set_position([pos1.x0, pos1.y0, pos3.width, pos3.height])
    # 在ax1上绘制中心线
    ax1.axvline(center_y, color='b')
    ax1.axhline(center_x, color='r')

    # 绘制垂直线强度分布图
    # set ax2 subtitle
    ax2.set_title('Vertical line intensity')
    ax2.plot(v_line_intensity, np.arange(data.shape[0])[::-1], color='b', label='Intensity')
    ax2.invert_yaxis()
    # 在ax2上进行高斯拟合
    popt, _ = curve_fit(gaussian, np.arange(data.shape[0]), v_line_intensity, p0=(center_x, 10, 100))
    fit_curve = gaussian(np.arange(data.shape[0]), *popt)
    half_max_val = popt[2] / 2
    half_max_indices = np.where(fit_curve >= half_max_val)[0]
    fwhm = (half_max_indices[-1] - half_max_indices[0]) * 2
    # 在ax2上绘制高斯拟合曲线和标注
    ax2.plot(fit_curve, np.arange(data.shape[0])[::-1], ls='--', color='b', label='Gaussian fit')
    fwhm = fwhm * 4.4
    ax2.axhline(y=half_max_indices[0], color='gray', linestyle='--')
    ax2.axhline(y=half_max_indices[-1], color='gray', linestyle='--')
    ax2.text(0.8, 0.8, f'FWHM: {fwhm:.2f} um', transform=ax2.transAxes, color='b')
    # set ax2 xlable: intensity, ylable: distance
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Distance')
    ax2.legend()
    # 绘制水平线强度分布图
    ax3.set_title('Horizontal line intensity')
    ax3.plot(np.arange(data.shape[1]), h_line_intensity, color='r', label='Intensity')
    # 在ax3上进行高斯拟合
    popt, _ = curve_fit(gaussian, np.arange(data.shape[1]), h_line_intensity, p0=(center_y, 10, 100))
    fit_curve = gaussian(np.arange(data.shape[1]), *popt)
    half_max_val = popt[2] / 2
    half_max_indices = np.where(fit_curve >= half_max_val)[0]
    fwhm = (half_max_indices[-1] - half_max_indices[0]) * 2
    # 在ax3上绘制高斯拟合曲线和标注
    ax3.plot(np.arange(data.shape[1]), fit_curve, ls='--', color='r', label='Gaussian fit')
    fwhm = fwhm * 4.4
    ax3.axvline(x=half_max_indices[0], color='gray', linestyle='--')
    ax3.axvline(x=half_max_indices[-1], color='gray', linestyle='--')
    ax3.text(0.8, 0.8, f'FWHM: {fwhm:.2f} um', transform=ax3.transAxes, color='r')
    # set ax3 xlable: distance, ylable: intensity
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Intensity')
    ax3.legend()
    ax4.axis('off')
    plt.savefig(args.output_path)
    plt.show()
