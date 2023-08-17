import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 输入数据
z = np.array([-0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015])  # 位置 (m)
FWHM = np.array([72.6, 61.6, 55, 50.6, 55, 57.2, 68.2])*2*1e-6  # FWHM (m)
FWHM = FWHM*1.73 # modified in 2023-08-17
lambda_ = 532e-9                                              # 光波长 (m)

# 定义高斯光束传播函数 (z: 位置，w0: 腰半径，zR: Rayleigh 范围)，返回光束半径
def w_z(z, w0, zR):
    w = w0 * np.sqrt(1 + (z / zR)**2)
    return w

# 定义 FWHM 和光束尺寸的转换函数
def FWHM_to_w(FWHM):
    w = FWHM/1.17 #* (2 * np.sqrt(2 * np.log(2)))/ np.pi
    return w
def w_to_FWHM(w):
    FWHM = w * 1.17 #np.pi / (2 * np.sqrt(2 * np.log(2)))
    return FWHM

# 将 FWHM 转化为光束尺寸
w = FWHM_to_w(FWHM)                                           # 光束半径 (m，注意单位转换)

# 初始参数猜测
w0_guess = 60e-6                                              # 初始腰半径估计值 (m)
# Rayleigh 范围计算 zR = pi * w0^2 / lambda 固定
zR_guess = np.pi * w0_guess**2 / lambda_                      # 初始 Rayleigh 范围估计值 (m)

# 拟合高斯光束传播曲线
popt, pcov = curve_fit(w_z, z, w, p0=[w0_guess, zR_guess])

# 输出拟合参数
w0_fit = popt[0]
zR_fit = popt[1]
print(f"Fitted w0 = {w0_fit*1e6:.2f} um, zR = {zR_fit*1e2:.2f} cm")
print(f"FWHM = {w_to_FWHM(w0_fit)*1e6:.2f} um")

w_fit = w_z(z, *popt)
# FWHM 和 光束尺寸的转换：FWHM = 2 * w * sqrt(2 * ln(2))
FWHM_fit = w_to_FWHM(w_fit) 

# 可视化FWHM结果
plt.figure(figsize=(8, 20))
plt.subplot(2, 1, 1)
z_plot = np.linspace(-0.02, 0.02, 500)
w_plot =  w_z(z_plot, *popt)
FWHM_plot = w_to_FWHM(w_plot)

plt.plot(z*1e2, FWHM*1e6, 'o', label="Data")  # 绘制数据点
plt.plot(z_plot*1e2, FWHM_plot*1e6, 'g--',label="Fit FWHM(z)")  # 绘制拟合曲线
plt.plot(z_plot*1e2, -1*FWHM_plot*1e6, 'g--')  # 绘制拟合曲线

# 绘制出x，y坐标轴
plt.plot([0, 0], [-1*max(FWHM_plot*1e6), max(FWHM_plot*1e6)], 'k-')
plt.plot([-1*max(z_plot*1e2), max(z_plot*1e2)], [0, 0], 'k-')

plt.xlabel("Position z (cm)")
plt.ylabel("um")
plt.legend()
plt.title("Gaussian beam propagation", loc='right')


# 可视化光束半径结果
plt.subplot(2, 1, 2)
# 绘制高斯光束传播曲线
z_plot = np.linspace(-0.02, 0.02, 500)
w_plot = w0_fit * np.sqrt(1 + (z_plot / zR_fit)**2)
plt.plot(z_plot*1e2, w_plot*1e6, 'b--', label="Gaussian beam curve\nFit w(z)")
plt.plot(z_plot*1e2, w_plot*(-1)*1e6, 'b--')

# 绘制出x，y坐标轴
plt.plot([0, 0], [-1*max(w_plot*1e6), max(w_plot*1e6)], 'k-')
plt.plot([-1*max(z_plot*1e2), max(z_plot*1e2)], [0, 0], 'k-')

plt.xlabel("Position z (cm)")
plt.ylabel("um")
# 图形设置,标题放在右上方
plt.title("Gaussian beam propagation", loc='right')
plt.legend()
plt.show()
