# -*- coding: utf-8 -*-
"""
超参数敏感性分析完整绘图脚本
数据：steps, eps, q, mean_accuracy, std_accuracy
生成四组关键图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
from io import StringIO

# ==================== 1. 数据加载 ====================
data = """
mean_accuracy,std_accuracy,steps,eps,q
47.4667,3.1944,40,0.01,10
50,1.5202,40,0.01,20
27.7333,5.1051,40,0.05,1
38,8.2408,40,0.05,5
46.4,5.7442,40,0.05,10
50.8,2.125,40,0.05,20
19.4667,2.8721,40,0.1,1
35.7333,7.3345,40,0.1,5
45.0667,6.4305,40,0.1,10
49.0667,4.4741,40,0.1,20
47.2,2.9635,80,0.01,10
46,3.1552,80,0.01,20
26.2667,8.8403,80,0.05,1
41.2,6.4965,80,0.05,5
47.6,1.8667,80,0.05,10
44.2667,4.3939,80,0.05,20
35.3333,5.1294,80,0.1,1
38.5333,7.1975,80,0.1,5
48.2667,2.2549,80,0.1,10
47.6,1.2365,80,0.1,20
49.2,7.4165,100,0.01,10
48.2667,4.3533,100,0.01,20
46.8,3.1098,100,0.05,10
46.2667,5.3433,100,0.05,20
50.2667,4.4741,100,0.1,10
48.2667,2.8158,100,0.1,20
"""

# df = pd.read_csv(StringIO(data))
df = pd.read_csv("output/summary.csv")
# 确保数据类型正确
df['steps'] = df['steps'].astype(int)
df['eps'] = df['eps'].astype(float)
df['q'] = df['q'].astype(int)

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用支持英文的字体

# ==================== 2. 图1：单参数敏感性（三个子图） ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) 对 eps 的敏感性 (固定 steps=40, q=10)
subset = df[(df['steps']==40) & (df['q']==10)]
axes[0].errorbar(subset['eps'], subset['mean_accuracy'], yerr=subset['std_accuracy'],
                 fmt='o-', capsize=5, color='royalblue', ecolor='gray', elinewidth=1)
axes[0].set_xlabel('eps', fontsize=12)
axes[0].set_ylabel('Mean Accuracy', fontsize=12)
axes[0].set_title('Sensitivity to eps\n(steps=40, q=10)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)

# (b) 对 q 的敏感性 (固定 eps=0.05, steps=40)
subset = df[(df['eps']==0.05) & (df['steps']==40)]
axes[1].errorbar(subset['q'], subset['mean_accuracy'], yerr=subset['std_accuracy'],
                 fmt='o-', capsize=5, color='seagreen', ecolor='gray', elinewidth=1)
axes[1].set_xlabel('q', fontsize=12)
axes[1].set_ylabel('Mean Accuracy', fontsize=12)
axes[1].set_title('Sensitivity to q\n(eps=0.05, steps=40)', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.6)

# (c) 对 steps 的敏感性 (固定 eps=0.1, q=10)
subset = df[(df['eps']==0.1) & (df['q']==10)]
axes[2].errorbar(subset['steps'], subset['mean_accuracy'], yerr=subset['std_accuracy'],
                 fmt='o-', capsize=5, color='darkorange', ecolor='gray', elinewidth=1)
axes[2].set_xlabel('steps', fontsize=12)
axes[2].set_ylabel('Mean Accuracy', fontsize=12)
axes[2].set_title('Sensitivity to steps\n(eps=0.1, q=10)', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('Fig1_single_param_sensitivity.svg',bbox_inches='tight')
plt.show()

# ==================== 3. 图2：双参数交互热力图 ====================
# 固定 steps 的不同取值，绘制 eps vs q 的 mean_accuracy 热力图
steps_values = [40, 80, 100]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, s in enumerate(steps_values):
    sub = df[df['steps'] == s]
    pivot = sub.pivot_table(values='mean_accuracy', index='eps', columns='q', aggfunc='mean')
    # 绘制热力图，缺失数据会显示为空白（默认灰色）
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', 
                cbar_kws={'label': 'Mean Accuracy'}, ax=axes[idx],
                linewidths=0.5, linecolor='white')
    axes[idx].set_title(f'steps = {s}', fontsize=12)
    axes[idx].set_xlabel('q', fontsize=11)
    axes[idx].set_ylabel('eps', fontsize=11)

plt.suptitle('Interaction between eps and q (fixed steps)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('Fig2_interaction_heatmaps.svg',  bbox_inches='tight')
plt.show()

# ==================== 4. 图3：并行坐标图（全局参数空间） ====================
# 将 mean_accuracy 分为三个等级：低(0-30), 中(30-45), 高(45-55)
df['acc_level'] = pd.cut(df['mean_accuracy'], bins=[0, 30, 45, 55], 
                         labels=['Low', 'Mid', 'High'])

# 对三个参数进行归一化，以便在同一轴上比较
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[['steps_norm', 'eps_norm', 'q_norm']] = scaler.fit_transform(df[['steps', 'eps', 'q']])

# 绘制并行坐标图
plt.figure(figsize=(8, 5))
parallel_coordinates(df_norm, class_column='acc_level', 
                     cols=['steps_norm', 'eps_norm', 'q_norm'],
                     colormap=plt.get_cmap('Set2'), alpha=0.7)
plt.xticks([0, 1, 2], ['steps', 'eps', 'q'], fontsize=12)
plt.ylabel('Normalized Parameter Value', fontsize=12)
plt.title('Parallel Coordinates: Parameter paths to different accuracy levels', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Accuracy Level', loc='upper right')
plt.tight_layout()
plt.savefig('Fig3_parallel_coordinates.svg', bbox_inches='tight')
plt.show()

# ==================== 5. 图4：均值 vs 标准差散点图（稳定性分析） ====================
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['mean_accuracy'], df['std_accuracy'], 
                      c=df['q'], cmap='plasma', s=80, alpha=0.8, edgecolors='black')
cbar = plt.colorbar(scatter)
cbar.set_label('q value', fontsize=11)
plt.xlabel('Mean Accuracy', fontsize=12)
plt.ylabel('Standard Deviation', fontsize=12)
plt.title('Accuracy vs Stability (color = q)', fontsize=13)

# 标注每个点对应的 (eps, q) 组合，避免重叠过多，可选标注前几个极值点
for i, row in df.iterrows():
    # 只标注高精度（>48）或低精度（<30）的点
    if row['mean_accuracy'] > 48 or row['mean_accuracy'] < 30:
        plt.annotate(f"({row['eps']},{row['q']})", 
                     (row['mean_accuracy'], row['std_accuracy']),
                     fontsize=8, xytext=(5,5), textcoords='offset points')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Fig4_accuracy_vs_stability.svg', bbox_inches='tight')
plt.show()

# ==================== 附：打印定量敏感性指标 ====================
print("========== 定量敏感性指标 ==========")

# 单参数敏感度计算示例（归一化范围）
def sensitivity(df, fixed_cond, param, value_col='mean_accuracy'):
    sub = df.copy()
    for k, v in fixed_cond.items():
        sub = sub[sub[k] == v]
    if len(sub) == 0:
        return None
    max_val = sub[value_col].max()
    min_val = sub[value_col].min()
    mean_val = sub[value_col].mean()
    return (max_val - min_val) / mean_val if mean_val != 0 else 0

sens_eps = sensitivity(df, {'steps':40, 'q':10}, 'eps')
sens_q = sensitivity(df, {'steps':40, 'eps':0.05}, 'q')
sens_steps = sensitivity(df, {'steps':100, 'eps':0.1, 'q':10}, 'steps')  # 需要调整，这里示例
print(f"灵敏度 (eps, fixed steps=40,q=10): {sens_eps:.3f}")
print(f"灵敏度 (q, fixed steps=40,eps=0.05): {sens_q:.3f}")
# steps 由于数据在固定 eps=0.1,q=10 时有三个点 (40,80,100)
sub_steps = df[(df['eps']==0.1) & (df['q']==10)]
if len(sub_steps)>=2:
    sens_steps = (sub_steps['mean_accuracy'].max() - sub_steps['mean_accuracy'].min()) / sub_steps['mean_accuracy'].mean()
    print(f"灵敏度 (steps, fixed eps=0.1,q=10): {sens_steps:.3f}")

# 最佳稳定组合
best_stable = df.loc[(df['mean_accuracy'] > 48) & (df['std_accuracy'] < 2.5)]
print("\n高精度且低标准差 (mean>48, std<2.5) 的组合:")
print(best_stable[['steps','eps','q','mean_accuracy','std_accuracy']])

print("\n脚本执行完成，所有图片已保存至当前目录。")