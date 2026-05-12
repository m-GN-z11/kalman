import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 读取 C++ 输出的滤波/预测数据
# ============================================================
data = np.loadtxt('kalman_output.txt', comments='#')

t       = data[:, 0]
true_cx = data[:, 1]
true_cy = data[:, 2]
meas_cx = data[:, 3]
meas_cy = data[:, 4]
filt_cx = data[:, 5]
filt_cy = data[:, 6]

true_vx = data[:, 7]
true_vy = data[:, 8]
filt_vx = data[:, 9]
filt_vy = data[:, 10]

true_area   = data[:, 11]
meas_area   = data[:, 12]
filt_area   = data[:, 13]
true_energy = data[:, 14]
meas_energy = data[:, 15]
filt_energy = data[:, 16]
is_forecast = data[:, 17].astype(bool)

# ============================================================
# 2. 生成完整的 100 个点的真实轨迹 (t = 0 ~ 10.0)
# ============================================================
t_full = np.arange(0, 10.01, 0.1)  # 0.0, 0.1, ..., 10.0

def true_cx_fn(t): return 10.0 + 1.6*t - 0.66*t**2 + 0.1*t**3 - 0.005*t**4
def true_cy_fn(t): return 20.0 + 2.0*t - 0.8*t**2 + 0.12*t**3 - 0.006*t**4
def true_vx_fn(t): return 1.6 - 1.32*t + 0.3*t**2 - 0.02*t**3
def true_vy_fn(t): return 2.0 - 1.6*t + 0.36*t**2 - 0.024*t**3
def true_area_fn(t): return 100.0 + 3.0*t - 1.5*t**2 + 0.2*t**3 - 0.01*t**4
def true_energy_fn(t): return 200.0 + 4.0*t - 2.0*t**2 + 0.3*t**3 - 0.015*t**4

true_cx_full    = true_cx_fn(t_full)
true_cy_full    = true_cy_fn(t_full)
true_vx_full    = true_vx_fn(t_full)
true_vy_full    = true_vy_fn(t_full)
true_area_full  = true_area_fn(t_full)
true_energy_full= true_energy_fn(t_full)

# ============================================================
# 3. 分离滤波点和预测点
# ============================================================
t_filt      = t[~is_forecast]
t_pred      = t[is_forecast]

filt_cx_f      = filt_cx[~is_forecast]
filt_cx_p      = filt_cx[is_forecast]
filt_cy_f      = filt_cy[~is_forecast]
filt_cy_p      = filt_cy[is_forecast]
filt_vx_f      = filt_vx[~is_forecast]
filt_vx_p      = filt_vx[is_forecast]
filt_vy_f      = filt_vy[~is_forecast]
filt_vy_p      = filt_vy[is_forecast]
filt_area_f    = filt_area[~is_forecast]
filt_area_p    = filt_area[is_forecast]
filt_energy_f  = filt_energy[~is_forecast]
filt_energy_p  = filt_energy[is_forecast]

meas_cx_f      = meas_cx[~is_forecast]
meas_cy_f      = meas_cy[~is_forecast]
meas_area_f    = meas_area[~is_forecast]
meas_energy_f  = meas_energy[~is_forecast]

# ============================================================
# 4. 绘图
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Kalman Filter: 50-point Filtering + 1-point Prediction', fontsize=14)

# ---------- Position cx ----------
ax = axes[0, 0]
ax.plot(t_full, true_cx_full, 'k-',  linewidth=1.5, label='True cx (full 100 pts)')
ax.scatter(t_filt, meas_cx_f, s=10, c='red', alpha=0.5, label='Meas cx')
ax.plot(t_filt, filt_cx_f,   'b--', linewidth=1.5, label='Filter cx')
if len(t_pred) > 0:
    ax.plot(t_pred, filt_cx_p, 'o', color='orange', markersize=8,
            markerfacecolor='orange', markeredgecolor='darkred', markeredgewidth=1.5,
            label='Predict cx')
ax.axvline(x=5.0, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
ax.set_title('Position cx')
ax.legend(fontsize=7, loc='best')
ax.grid(True)

# ---------- Position cy ----------
ax = axes[0, 1]
ax.plot(t_full, true_cy_full, 'k-',  linewidth=1.5, label='True cy (full 100 pts)')
ax.scatter(t_filt, meas_cy_f, s=10, c='red', alpha=0.5, label='Meas cy')
ax.plot(t_filt, filt_cy_f,   'b--', linewidth=1.5, label='Filter cy')
if len(t_pred) > 0:
    ax.plot(t_pred, filt_cy_p, 'o', color='orange', markersize=8,
            markerfacecolor='orange', markeredgecolor='darkred', markeredgewidth=1.5,
            label='Predict cy')
ax.axvline(x=5.0, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
ax.set_title('Position cy')
ax.legend(fontsize=7, loc='best')
ax.grid(True)

# ---------- Velocity vx ----------
ax = axes[0, 2]
ax.plot(t_full, true_vx_full, 'k-',  linewidth=1.5, label='True vx (full 100 pts)')
ax.plot(t_filt, filt_vx_f,   'b--', linewidth=1.5, label='Filter vx')
if len(t_pred) > 0:
    ax.plot(t_pred, filt_vx_p, 'o', color='orange', markersize=8,
            markerfacecolor='orange', markeredgecolor='darkred', markeredgewidth=1.5,
            label='Predict vx')
ax.axvline(x=5.0, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
ax.set_title('Velocity vx')
ax.legend(fontsize=7, loc='best')
ax.grid(True)

# ---------- Velocity vy ----------
ax = axes[1, 0]
ax.plot(t_full, true_vy_full, 'k-',  linewidth=1.5, label='True vy (full 100 pts)')
ax.plot(t_filt, filt_vy_f,   'b--', linewidth=1.5, label='Filter vy')
if len(t_pred) > 0:
    ax.plot(t_pred, filt_vy_p, 'o', color='orange', markersize=8,
            markerfacecolor='orange', markeredgecolor='darkred', markeredgewidth=1.5,
            label='Predict vy')
ax.axvline(x=5.0, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
ax.set_title('Velocity vy')
ax.legend(fontsize=7, loc='best')
ax.grid(True)

# ---------- Area ----------
ax = axes[1, 1]
ax.plot(t_full, true_area_full, 'k-',  linewidth=1.5, label='True area (full 100 pts)')
ax.scatter(t_filt, meas_area_f, s=10, c='red', alpha=0.5, label='Meas area')
ax.plot(t_filt, filt_area_f,   'b--', linewidth=1.5, label='Filter area')
if len(t_pred) > 0:
    ax.plot(t_pred, filt_area_p, 'o', color='orange', markersize=8,
            markerfacecolor='orange', markeredgecolor='darkred', markeredgewidth=1.5,
            label='Predict area')
ax.axvline(x=5.0, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
ax.set_title('Area')
ax.legend(fontsize=7, loc='best')
ax.grid(True)

# ---------- Energy ----------
ax = axes[1, 2]
ax.plot(t_full, true_energy_full, 'k-',  linewidth=1.5, label='True energy (full 100 pts)')
ax.scatter(t_filt, meas_energy_f, s=10, c='red', alpha=0.5, label='Meas energy')
ax.plot(t_filt, filt_energy_f,   'b--', linewidth=1.5, label='Filter energy')
if len(t_pred) > 0:
    ax.plot(t_pred, filt_energy_p, 'o', color='orange', markersize=8,
            markerfacecolor='orange', markeredgecolor='darkred', markeredgewidth=1.5,
            label='Predict energy')
ax.axvline(x=5.0, color='gray', linestyle=':', alpha=0.7, label='Forecast start')
ax.set_title('Energy')
ax.legend(fontsize=7, loc='best')
ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('kalman_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved to kalman_result.png")