# -*- coding: utf-8 -*-
"""
超参数敏感性分析完整绘图脚本（已添加坐标轴单位）
数据：steps, eps, q, mean_accuracy (%), std_accuracy (%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
from io import StringIO

# ==================== 1. 数据加载 ====================

df = pd.read_csv("output/summary.csv")

df['steps'] = df['steps'].astype(int)
df['eps'] = df['eps'].astype(float)
df['q'] = df['q'].astype(int)

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 1.5

# ==================== 图1a: eps 敏感性（对数坐标） ====================
plt.figure(figsize=(6, 4.5))
subset = df[(df['steps']==40) & (df['q']==20)].sort_values('eps') 
plt.errorbar(subset['eps'], subset['mean_accuracy'], yerr=subset['std_accuracy'],
             fmt='o-', capsize=5, color='royalblue', ecolor='gray', elinewidth=1)
plt.xscale('log')   # 关键修改
plt.xlabel('eps', fontsize=12)
plt.ylabel('Mean Accuracy (%)', fontsize=12)
plt.title('Sensitivity to eps (steps=40, q=20)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig1a_sensitivity_eps.svg', bbox_inches='tight')
plt.show()
# # ==================== 图1a: eps 敏感性 ====================
# plt.figure(figsize=(6, 4.5))
# subset = df[(df['steps']==40) & (df['q']==10)].sort_values('eps') 
# plt.errorbar(subset['eps'], subset['mean_accuracy'], yerr=subset['std_accuracy'],
#              fmt='o-', capsize=5, color='royalblue', ecolor='gray', elinewidth=1)
# plt.xlabel('eps', fontsize=12)
# plt.ylabel('Mean Accuracy (%)', fontsize=12)   # 加单位
# plt.title('Sensitivity to eps (steps=40, q=10)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig('Fig1a_sensitivity_eps.svg', bbox_inches='tight')
# plt.show()

# ==================== 图1b: q 敏感性 ====================
plt.figure(figsize=(6, 4.5))
subset = df[(df['eps']==0.01) & (df['steps']==40)].sort_values('q') 
plt.errorbar(subset['q'], subset['mean_accuracy'], yerr=subset['std_accuracy'],
             fmt='o-', capsize=5, color='seagreen', ecolor='gray', elinewidth=1)
plt.xlabel('q', fontsize=12)
plt.ylabel('Mean Accuracy (%)', fontsize=12)   # 加单位
plt.title('Sensitivity to q (eps=0.01, steps=40)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig1b_sensitivity_q.svg', bbox_inches='tight')
plt.show()

# ==================== 添加新数据点 ====================
new_row = pd.DataFrame({
    'steps': [20],
    'eps': [0.01],
    'q': [20],
    'mean_accuracy': [37.1],
    'std_accuracy': [3.0]
})
df = pd.concat([df, new_row], ignore_index=True)

# 确保数据类型正确（steps 已经是 int，但新加的可能为 float，强制转换）
df['steps'] = df['steps'].astype(int)
df['eps'] = df['eps'].astype(float)
df['q'] = df['q'].astype(int)

# ==================== 图1c: steps 敏感性 ====================
plt.figure(figsize=(6, 4.5))
subset = df[(df['eps']==0.01) & (df['q']==20)].sort_values('steps') 
plt.errorbar(subset['steps'], subset['mean_accuracy'], yerr=subset['std_accuracy'],
             fmt='o-', capsize=5, color='darkorange', ecolor='gray', elinewidth=1)
plt.xlabel('steps (iterations)', fontsize=12)
plt.ylabel('Mean Accuracy (%)', fontsize=12)
plt.title('Sensitivity to steps (eps=0.01, q=20)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig1c_sensitivity_steps.svg', bbox_inches='tight')
plt.show()

# ==================== 图2: 热力图 ====================
steps_values = [40, 80]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for idx, s in enumerate(steps_values):
    sub = df[df['steps'] == s]
    pivot = sub.pivot_table(values='mean_accuracy', index='eps', columns='q', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis',
                cbar_kws={'label': 'Mean Accuracy (%)'}, ax=axes[idx],   # colorbar 加单位
                linewidths=0.5, linecolor='white')
    axes[idx].set_title(f'steps = {s}', fontsize=12)
    axes[idx].set_xlabel('q', fontsize=11)
    axes[idx].set_ylabel('eps', fontsize=11)

plt.suptitle('Interaction between eps and q (fixed steps)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('Fig2_interaction_heatmaps.svg', bbox_inches='tight')
plt.show()

# ==================== 图3: 并行坐标图 ====================
df['acc_level'] = pd.cut(df['mean_accuracy'], bins=[0, 30, 45, 55],
                         labels=['Low', 'Mid', 'High'])

scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[['steps_norm', 'eps_norm', 'q_norm']] = scaler.fit_transform(df[['steps', 'eps', 'q']])

plt.figure(figsize=(8, 5))
parallel_coordinates(df_norm, class_column='acc_level',
                     cols=['steps_norm', 'eps_norm', 'q_norm'],
                     colormap=plt.get_cmap('Set2'), alpha=0.7)
plt.xticks([0, 1, 2], ['steps', 'eps', 'q'], fontsize=12)
plt.ylabel('Normalized Parameter Value (0-1)', fontsize=12)   # 明确无量纲
plt.title('Parallel Coordinates: Parameter paths to different accuracy levels', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Accuracy Level', loc='upper right')
plt.tight_layout()
plt.savefig('Fig3_parallel_coordinates.svg', bbox_inches='tight')
plt.show()

# ==================== 图4: 稳定性散点图 ====================
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['mean_accuracy'], df['std_accuracy'],
                      c=df['q'], cmap='plasma', s=80, alpha=0.8, edgecolors='black')
cbar = plt.colorbar(scatter)
cbar.set_label('q value', fontsize=11)
plt.xlabel('Mean Accuracy (%)', fontsize=12)      # 加单位
plt.ylabel('Standard Deviation (%)', fontsize=12) # 加单位
plt.title('Accuracy vs Stability (color = q)', fontsize=13)

for i, row in df.iterrows():
    if row['mean_accuracy'] > 48 or row['mean_accuracy'] < 30:
        plt.annotate(f"({row['eps']},{row['q']})",
                     (row['mean_accuracy'], row['std_accuracy']),
                     fontsize=8, xytext=(5,5), textcoords='offset points')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Fig4_accuracy_vs_stability.svg', bbox_inches='tight')
plt.show()

# ==================== 定量指标打印 ====================
print("========== 定量敏感性指标 ==========")

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
print(f"灵敏度 (eps, fixed steps=40,q=10): {sens_eps:.3f}")
print(f"灵敏度 (q, fixed steps=40,eps=0.05): {sens_q:.3f}")

sub_steps = df[(df['eps']==0.1) & (df['q']==10)]
if len(sub_steps)>=2:
    sens_steps = (sub_steps['mean_accuracy'].max() - sub_steps['mean_accuracy'].min()) / sub_steps['mean_accuracy'].mean()
    print(f"灵敏度 (steps, fixed eps=0.1,q=10): {sens_steps:.3f}")

best_stable = df.loc[(df['mean_accuracy'] > 48) & (df['std_accuracy'] < 2.5)]
print("\n高精度且低标准差 (mean>48%, std<2.5%) 的组合:")
print(best_stable[['steps','eps','q','mean_accuracy','std_accuracy']])

print("\n脚本执行完成，所有图片已保存至当前目录。")