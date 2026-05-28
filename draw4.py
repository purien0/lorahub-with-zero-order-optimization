import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 原始数据：名称，组合权重，单独准确率(%)
data = [
    ("glue_mrpc", 0.6348, 15.3),
    ("dream_baseline", 1.8563, 19.3),
    ("ropes_given_background_situation", -0.2499, 14.6),
    ("quarel_logic_test", -0.0205, 16.0),
    ("anli_r1", -0.0649, 18.6),
    ("imdb_reviews_plain_text", 0.2791, 16.6),
    ("race_high_Is_this_the_right_answer", 1.8147, 19.3),
    ("ropes_plain_no_background", -0.0402, 14.6),
    ("duorc_SelfRC_movie_director", 0.2362, 14.6),
    ("quail_context_question_description_answer_text", 1.0862, 14.6),
    ("kilt_tasks_hotpotqa_final_exam", -0.2585, 15.3),
    ("quartz_read_passage_below_choose", -0.0484, 15.3),
    ("wiki_hop_original_generate_subject_and_object", -0.0095, 14.6),
    ("wiki_hop_original_choose_best_object_interrogative_2", -0.1133, 16.6),
    ("wiki_qa_exercise", 0.3999, 15.3),
    ("wiki_hop_original_choose_best_object_affirmative_3", -0.0169, 14.6),
    ("duorc_SelfRC_title_generation", -0.0753, 16.6),
    ("true_case", -0.9839, 9.3),
    ("adversarial_qa_dbert_question_context_answer", -0.3399, 16.0),
    ("super_glue_wsc.fixed", -0.3667, 17.3)
]

df = pd.DataFrame(data, columns=["module", "weight", "accuracy"])

# 按权重降序排序，取前15个作为 Top-K（可根据需要改为全部）
df_sorted = df.sort_values("weight", ascending=False).head(15)

# 创建双轴图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 条形图：权重（左轴）
x = np.arange(len(df_sorted))
bars = ax1.bar(x, df_sorted["weight"], color='steelblue', alpha=0.7, label='组合权重')
ax1.set_xlabel("LoRA 模块")
ax1.set_ylabel("组合权重", color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(x)
ax1.set_xticklabels(df_sorted["module"], rotation=45, ha='right', fontsize=9)

# 折线图：准确率（右轴）
ax2 = ax1.twinx()
line = ax2.plot(x, df_sorted["accuracy"], color='darkred', marker='o', linestyle='-', linewidth=2, label='单独准确率 (%)')
ax2.set_ylabel("单独准确率 (%)", color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')

# 添加数值标签（在条形上方）
for i, (w, acc) in enumerate(zip(df_sorted["weight"], df_sorted["accuracy"])):
    ax1.text(i, w + 0.05, f"{w:.2f}", ha='center', va='bottom', fontsize=8, color='steelblue')
    ax2.text(i, acc + 0.5, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8, color='darkred')

# 图例和标题
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Top-15 组合权重与对应模块单独准确率分布")
plt.tight_layout()
plt.savefig("topk_weight_accuracy.png", dpi=300)
plt.show()


plt.figure(figsize=(10, 6))
colors = np.where(df["weight"] > 0, 'green', 'red')
plt.scatter(df["weight"], df["accuracy"], c=colors, alpha=0.7, s=80)
for _, row in df.iterrows():
    plt.annotate(row["module"], (row["weight"], row["accuracy"]), fontsize=7, ha='center')
plt.axhline(y=df["accuracy"].mean(), linestyle='--', color='gray', alpha=0.5)
plt.axvline(x=0, linestyle='--', color='black', alpha=0.5)
plt.xlabel("组合权重")
plt.ylabel("单独准确率 (%)")
plt.title("LoRA 模块的权重-准确率分布（全部20个）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("weight_accuracy_scatter.png", dpi=300)
plt.show()