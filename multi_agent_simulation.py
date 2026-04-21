"""
多智能体编队控制仿真
合作-对抗关系网络上的多智能体编队控制

系统模型 (N=4 双积分器智能体):
  y_i'' = u_i
  
控制律 (对应论文中 u_i = K1i*hat_x_i + d_i*K2i*xi_i - Bi^{-1}Di*hat_di):
  u_i = K1*(d_i*xi_i1 - y_i) + K2*(d_i*xi_i2 - y_i')   [忽略扰动项]

内部模型 (对应 xi_i_dot = M*xi_i + omega*sum(a_ij*(d_j*y_j - d_i*y_i))):
  xi_i1_dot = xi_i2 + c1 * sum_{j in Ni} a_ij*(d_j*y_j - d_i*y_i)
  xi_i2_dot = -omega0^2 * xi_i1
  其中 omega = [c1; 0]^T (注入向量，注入第一方程以保证稳定性)

通信拓扑:
  邻接矩阵 A (a_ij >= 0，连接强度)
  D = diag(d_1,...,d_N), d_i in {-1,+1} (编码合作/对抗关系)
  结构平衡: {1,3} 为第一团体(d=+1), {2,4} 为第二团体(d=-1)

稳定性:
  不一致性模式特征多项式: s^3 + K1*s^2 + (omega0^2 + K1*c1*lambda)*s + K1*omega0^2
  由 Routh-Hurwitz 判据, 对任意 K1,K2,c1,lambda > 0 均稳定。
  一致性模式: 特征根为 -K1 (稳定) 和 ±j*omega0 (纯虚, 对应稳态振荡)。
  双积分器 PD 跟踪极点: s = -K2/2 ± j*sqrt(K1 - K2^2/4) (欠阻尼稳定)。
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 中文字体支持
# -------------------------------------------------------------------
matplotlib.rcParams['axes.unicode_minus'] = False
try:
    _fm = fm.FontManager()
    _cjk = [f.name for f in _fm.ttflist
            if any(k in f.name for k in
                   ['Zen Hei', 'WenQuan', 'YaHei', 'SimHei',
                    'Noto Sans CJK', 'Source Han'])]
    if _cjk:
        matplotlib.rcParams['font.sans-serif'] = \
            [_cjk[0]] + matplotlib.rcParams.get('font.sans-serif', [])
except Exception:
    pass

# -------------------------------------------------------------------
# 系统参数
# -------------------------------------------------------------------
N      = 4        # 智能体数量
omega0 = 1.0      # 参考信号角频率 (rad/s)，周期 T ≈ 6.28 s
K1     = 5.0      # 位置反馈增益 (K_{1i} = K1)
K2     = 3.0      # 速度反馈增益 (K_{2i} = K2)
c1     = 0.2      # 内部模型耦合增益 (注入向量第一分量)

# 符号矩阵 D = diag(d_1,...,d_N)
# 团体1: 智能体1,3 (下标0,2)  d=+1
# 团体2: 智能体2,4 (下标1,3)  d=-1
d = np.array([1.0, -1.0, 1.0, -1.0])

# 邻接矩阵 (a_ij >= 0，全连接拓扑)
# 合作/对抗关系由 d_i 编码：d_i*d_j=+1 为合作，d_i*d_j=-1 为对抗
A_adj = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
], dtype=float)

# -------------------------------------------------------------------
# ODE 右端函数
# 状态顺序: 每个智能体 i 对应 [y_i, vy_i, xi_i1, xi_i2]
# z = [y0,vy0,xi01,xi02,  y1,vy1,xi11,xi12,  y2,vy2,xi21,xi22,  y3,vy3,xi31,xi32]
# -------------------------------------------------------------------
def rhs(t, z):
    dz  = np.zeros(4 * N)
    y   = z[0::4]   # 位置 / 输出
    vy  = z[1::4]   # 速度
    xi1 = z[2::4]   # 内部模型状态1
    xi2 = z[3::4]   # 内部模型状态2

    for i in range(N):
        # 耦合项: sum_j a_ij*(d_j*y_j - d_i*y_i)
        # 等价展开: np.dot(A_adj[i], d*y) - d[i]*y[i]*sum(A_adj[i])
        coupling = np.dot(A_adj[i], d * y - d[i] * y[i])

        # 内部模型 (注入第一方程 — 保证稳定)
        dxi1_i = xi2[i] + c1 * coupling
        dxi2_i = -(omega0 ** 2) * xi1[i]

        # 控制律
        u_i = K1 * (d[i] * xi1[i] - y[i]) + K2 * (d[i] * xi2[i] - vy[i])

        # 双积分器动力学
        dz[4 * i]     = vy[i]
        dz[4 * i + 1] = u_i
        dz[4 * i + 2] = dxi1_i
        dz[4 * i + 3] = dxi2_i

    return dz

# -------------------------------------------------------------------
# 初始条件
# 设计原则:
#   avg(xi_i1) = 3  →  稳态振幅约为 3
#   智能体3(蓝,d=1): xi1=9,xi2=5  → u3(0)=50  → 出现约8的正峰值
#   智能体4(黑,d=-1): xi1=-7,xi2=-5 → u4(0)=50 → 出现约8的正峰值
#   智能体2(绿,d=-1): xi1=6,xi2=4  → u2(0)=-52 → 出现约-6的谷值
#   智能体1(红,d=1): xi1=4,xi2=2  → u1(0)=16  → 中等正峰值
#   avg xi1 = (4+6+9-7)/4 = 3 ✓
# -------------------------------------------------------------------
z0 = np.array([
    # y,   vy,   xi1,  xi2
    2.0,  0.0,  4.0,  2.0,   # 智能体1 (红,  d=+1)
    2.0,  0.0,  6.0,  4.0,   # 智能体2 (绿,  d=-1)
    2.0,  0.0,  9.0,  5.0,   # 智能体3 (蓝,  d=+1) — 大初值
    0.0,  0.0, -7.0, -5.0,   # 智能体4 (黑,  d=-1) — 大负初值
])

# -------------------------------------------------------------------
# 数值积分
# -------------------------------------------------------------------
t_span = (0.0, 50.0)
t_eval = np.linspace(0.0, 50.0, 5000)

print("正在运行仿真...")
sol = solve_ivp(rhs, t_span, z0, method='RK45',
                t_eval=t_eval, rtol=1e-8, atol=1e-10)

if not sol.success:
    raise RuntimeError(f"积分失败: {sol.message}")
print("仿真完成.")

# -------------------------------------------------------------------
# 绘图
# -------------------------------------------------------------------
colors_plot = ['red', 'green', 'blue', 'black']
labels_cn   = ['智能体1输出', '智能体2输出', '智能体3输出', '智能体4输出']

fig, ax = plt.subplots(figsize=(10, 6))

for i in range(N):
    y_out = sol.y[4 * i]
    ax.plot(sol.t, y_out, color=colors_plot[i],
            linewidth=1.5, label=labels_cn[i])

ax.set_xlabel('时间 (秒)', fontsize=12)
ax.set_ylabel('输出 y_i', fontsize=12)
ax.set_title('多智能体系统输出', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.4)
ax.set_xlim([0, 50])
ax.set_ylim([-7, 9])

plt.tight_layout()
out_file = 'multi_agent_output.png'
plt.savefig(out_file, dpi=120, bbox_inches='tight')
print(f"图像已保存: {out_file}")
plt.close()


