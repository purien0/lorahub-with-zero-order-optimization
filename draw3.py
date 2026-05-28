import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# 你的日志数据（直接粘贴）
log_data = """
[ZO-Adam] step=1, loss=0.128127,accuracy:15.333333
[ZO-Adam] step=2, loss=0.112404,accuracy:17.333333
[ZO-Adam] step=3, loss=0.112685,accuracy:16.666667
[ZO-Adam] step=4, loss=0.116840,accuracy:14.000000
[ZO-Adam] step=5, loss=0.117401,accuracy:15.333333
[ZO-Adam] step=6, loss=0.116336,accuracy:16.666667
[ZO-Adam] step=7, loss=0.111347,accuracy:18.000000
[ZO-Adam] step=8, loss=0.103440,accuracy:20.000000
[ZO-Adam] step=9, loss=0.093683,accuracy:22.666667
[ZO-Adam] step=10, loss=0.082127,accuracy:27.333333
[ZO-Adam] step=11, loss=0.072505,accuracy:32.666667
[ZO-Adam] step=12, loss=0.068455,accuracy:34.000000
[ZO-Adam] step=13, loss=0.071022,accuracy:34.000000
[ZO-Adam] step=14, loss=0.072217,accuracy:34.666667
[ZO-Adam] step=15, loss=0.068992,accuracy:35.333333
[ZO-Adam] step=16, loss=0.062259,accuracy:36.666667
[ZO-Adam] step=17, loss=0.058795,accuracy:37.333333
[ZO-Adam] step=18, loss=0.058833,accuracy:38.000000
[ZO-Adam] step=19, loss=0.059860,accuracy:38.000000
[ZO-Adam] step=20, loss=0.059670,accuracy:38.666667
[ZO-Adam] step=21, loss=0.057816,accuracy:40.000000
[ZO-Adam] step=22, loss=0.054826,accuracy:42.666667
[ZO-Adam] step=23, loss=0.055334,accuracy:42.000000
[ZO-Adam] step=24, loss=0.053364,accuracy:42.000000
[ZO-Adam] step=25, loss=0.049455,accuracy:42.000000
[ZO-Adam] step=26, loss=0.048203,accuracy:42.666667
[ZO-Adam] step=27, loss=0.047648,accuracy:44.666667
[ZO-Adam] step=28, loss=0.047275,accuracy:44.666667
[ZO-Adam] step=29, loss=0.046682,accuracy:46.000000
[ZO-Adam] step=30, loss=0.045519,accuracy:46.666667
[ZO-Adam] step=31, loss=0.043853,accuracy:46.000000
[ZO-Adam] step=32, loss=0.041765,accuracy:45.333333
[ZO-Adam] step=33, loss=0.039794,accuracy:44.000000
[ZO-Adam] step=34, loss=0.039242,accuracy:40.666667
[ZO-Adam] step=35, loss=0.040519,accuracy:41.333333
[ZO-Adam] step=36, loss=0.038032,accuracy:46.000000
[ZO-Adam] step=37, loss=0.041505,accuracy:49.333333
[ZO-Adam] step=38, loss=0.044266,accuracy:48.666667
[ZO-Adam] step=39, loss=0.045674,accuracy:50.000000
[ZO-Adam] step=40, loss=0.045826,accuracy:49.333333
[ZO-Adam] step=41, loss=0.044571,accuracy:49.333333
[ZO-Adam] step=42, loss=0.041686,accuracy:48.666667
[ZO-Adam] step=43, loss=0.038564,accuracy:48.000000
[ZO-Adam] step=44, loss=0.036635,accuracy:46.666667
[ZO-Adam] step=45, loss=0.039633,accuracy:45.333333
[ZO-Adam] step=46, loss=0.036961,accuracy:46.000000
[ZO-Adam] step=47, loss=0.035129,accuracy:49.333333
[ZO-Adam] step=48, loss=0.036971,accuracy:50.666667
[ZO-Adam] step=49, loss=0.035812,accuracy:50.000000
[ZO-Adam] step=50, loss=0.035659,accuracy:49.333333
[ZO-Adam] step=51, loss=0.035542,accuracy:49.333333
[ZO-Adam] step=52, loss=0.035281,accuracy:49.333333
[ZO-Adam] step=53, loss=0.034800,accuracy:48.666667
[ZO-Adam] step=54, loss=0.034255,accuracy:48.666667
[ZO-Adam] step=55, loss=0.033469,accuracy:49.333333
[ZO-Adam] step=56, loss=0.032605,accuracy:47.333333
[ZO-Adam] step=57, loss=0.031693,accuracy:49.333333
[ZO-Adam] step=58, loss=0.031017,accuracy:49.333333
[ZO-Adam] step=59, loss=0.030498,accuracy:47.333333
[ZO-Adam] step=60, loss=0.029869,accuracy:46.666667
[ZO-Adam] step=61, loss=0.029398,accuracy:46.000000
[ZO-Adam] step=62, loss=0.028966,accuracy:46.666667
[ZO-Adam] step=63, loss=0.028507,accuracy:46.000000
[ZO-Adam] step=64, loss=0.028030,accuracy:46.000000
[ZO-Adam] step=65, loss=0.027598,accuracy:46.000000
[ZO-Adam] step=66, loss=0.027361,accuracy:44.666667
[ZO-Adam] step=67, loss=0.027533,accuracy:45.333333
[ZO-Adam] step=68, loss=0.028226,accuracy:44.666667
[ZO-Adam] step=69, loss=0.027607,accuracy:44.666667
[ZO-Adam] step=70, loss=0.026641,accuracy:44.666667
[ZO-Adam] step=71, loss=0.026630,accuracy:46.000000
[ZO-Adam] step=72, loss=0.026729,accuracy:46.666667
[ZO-Adam] step=73, loss=0.026587,accuracy:47.333333
[ZO-Adam] step=74, loss=0.026224,accuracy:45.333333
[ZO-Adam] step=75, loss=0.025951,accuracy:44.000000
[ZO-Adam] step=76, loss=0.025617,accuracy:43.333333
[ZO-Adam] step=77, loss=0.025264,accuracy:44.000000
[ZO-Adam] step=78, loss=0.024676,accuracy:47.333333
[ZO-Adam] step=79, loss=0.024394,accuracy:47.333333
[ZO-Adam] step=80, loss=0.024100,accuracy:46.666667
[ZO-Adam] step=81, loss=0.023817,accuracy:47.333333
[ZO-Adam] step=82, loss=0.023550,accuracy:46.666667
[ZO-Adam] step=83, loss=0.023497,accuracy:48.000000
[ZO-Adam] step=84, loss=0.023684,accuracy:46.666667
[ZO-Adam] step=85, loss=0.023868,accuracy:46.666667
[ZO-Adam] step=86, loss=0.023486,accuracy:46.666667
[ZO-Adam] step=87, loss=0.023069,accuracy:47.333333
[ZO-Adam] step=88, loss=0.022972,accuracy:47.333333
[ZO-Adam] step=89, loss=0.022948,accuracy:47.333333
[ZO-Adam] step=90, loss=0.023044,accuracy:47.333333
[ZO-Adam] step=91, loss=0.023081,accuracy:49.333333
[ZO-Adam] step=92, loss=0.022936,accuracy:50.666667
[ZO-Adam] step=93, loss=0.022737,accuracy:50.666667
[ZO-Adam] step=94, loss=0.022634,accuracy:48.666667
[ZO-Adam] step=95, loss=0.022507,accuracy:46.666667
[ZO-Adam] step=96, loss=0.022362,accuracy:47.333333
[ZO-Adam] step=97, loss=0.022192,accuracy:51.333333
[ZO-Adam] step=98, loss=0.022060,accuracy:50.000000
[ZO-Adam] step=99, loss=0.021955,accuracy:50.000000
[ZO-Adam] step=100, loss=0.021851,accuracy:49.333333
"""

# 解析
steps, losses, accuracies = [], [], []
for line in StringIO(log_data):
    if 'step=' not in line:
        continue
    # 提取 step
    step_part = line.split('step=')[1].split(',')[0]
    step = int(step_part)
    # 提取 loss
    loss_part = line.split('loss=')[1].split(',')[0]
    loss = float(loss_part)
    # 提取 accuracy
    acc_part = line.split('accuracy:')[1].strip()
    acc = float(acc_part)
    steps.append(step)
    losses.append(loss)
    accuracies.append(acc)

# 创建双轴图
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左轴：Loss（红色曲线，黑色标签/刻度）
color1 = 'black'
ax1.set_xlabel('Iteration Steps', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12, color=color1)
ax1.plot(steps, losses, color='tab:red', linewidth=1.5, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.3)

# 右轴：Accuracy
ax2 = ax1.twinx()
color2 = 'black'
ax2.set_ylabel('Validation Accuracy (%)', fontsize=12, color=color2)

# --- 构建完整的平滑曲线（两端保留原始值）---
window = 5
# 计算有效平滑部分（长度 = len(accuracies) - window + 1）
convolved = np.convolve(accuracies, np.ones(window)/window, mode='valid')
# 初始化完整平滑数组为原始值
acc_smoothed_full = np.array(accuracies, dtype=float)
# 覆盖中间部分：窗口为奇数，偏移 offset = (window-1)//2 = 2
offset = (window - 1) // 2
for i in range(len(convolved)):
    acc_smoothed_full[i + offset] = convolved[i]
# 绘制完整的平滑曲线（两端会显示原始值，中间平滑）
ax2.plot(steps, acc_smoothed_full, color='tab:blue', linewidth=2, label='Smoothed Accuracy')
# -----------------------------------------------

# 标注 step=40, 80, 100 的点（使用原始准确率）
highlight_steps = [40, 80, 100]
for step in highlight_steps:
    idx = steps.index(step)
    acc_value = accuracies[idx]
    ax2.plot(step, acc_value, 'o', color='darkorange', markersize=8, markeredgecolor='black', markeredgewidth=0.8)
    ax2.annotate(f'{acc_value:.1f}%', (step, acc_value),
                 xytext=(5, 5 if step != 100 else -10),
                 textcoords='offset points', fontsize=9, ha='left')

# # 标注最佳准确率点
# max_acc_step = steps[accuracies.index(max(accuracies))]
# max_acc = max(accuracies)
# ax2.plot(max_acc_step, max_acc, 'ro', markersize=6, markeredgecolor='black')
# ax2.annotate(f'Best Acc: {max_acc:.1f}%', (max_acc_step, max_acc),
#              xytext=(10, -10), textcoords='offset points', fontsize=9, ha='left')

plt.title('Loss-accuracy discrepancy: Loss decreases while accuracy saturates and oscillates', fontsize=13)

# 合并图例，放在左上角，带背景框
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
           framealpha=0.8, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig('loss_accuracy_curve_highlight.svg', bbox_inches='tight')
plt.show()