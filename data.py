import os
import re
import glob
import pandas as pd


def parse_filename(name):
    """
    从文件名解析参数
    """
    pattern = r"multiseed_step(\d+)_eps([0-9.]+)_lr([0-9.]+)_q(\d+)"
    match = re.search(pattern, name)
    if match:
        return {
            "steps": int(match.group(1)),
            "eps": float(match.group(2)),
            "lr": float(match.group(3)),
            "q": int(match.group(4)),
        }
    return {}


def parse_log(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    name = os.path.basename(file_path)

    # ===== final loss（取最后一个）=====
    losses = re.findall(r"\[ori\] final loss=([0-9.]+)", text)
    final_loss = float(losses[-1]) if losses else None

    # ===== mean / std =====
    mean_match = re.search(r"mean accuracy:\s*([0-9.]+)", text)
    std_match = re.search(r"std accuracy:\s*([0-9.]+)", text)

    mean = float(mean_match.group(1)) if mean_match else None
    std = float(std_match.group(1)) if std_match else None

    # ===== 文件名参数 =====
    params = parse_filename(name)

    return {
        "file": name,
        "final_loss": final_loss,
        "mean_accuracy": mean,
        "std_accuracy": std,
        **params
    }
def analyze_fixed_eps(df):
    eps_values = sorted(df["eps"].unique())

    for eps in eps_values:
        sub = df[df["eps"] == eps]

        print(f"\n===== eps = {eps} =====")

        # 按 mean accuracy 排序
        sub_sorted = sub.sort_values(by="mean_accuracy", ascending=False)

        print(sub_sorted[["steps", "q", "mean_accuracy", "std_accuracy"]].to_string(index=False))
def analyze_fixed_q(df):
    q_values = sorted(df["q"].unique())

    for q in q_values:
        sub = df[df["q"] == q]

        print(f"\n===== q = {q} =====")

        sub_sorted = sub.sort_values(by="mean_accuracy", ascending=False)

        print(sub_sorted[["eps", "steps", "mean_accuracy", "std_accuracy"]].to_string(index=False))
def analyze_fixed_steps(df):
    steps_values = sorted(df["steps"].unique())

    for s in steps_values:
        sub = df[df["steps"] == s]

        print(f"\n===== steps = {s} =====")

        sub_sorted = sub.sort_values(by="mean_accuracy", ascending=False)

        print(sub_sorted[["eps", "q", "mean_accuracy", "std_accuracy"]].to_string(index=False))
        
def pivot_eps(df, eps_value):
    sub = df[df["eps"] == eps_value]

    pivot = sub.pivot_table(
        index="steps",
        columns="q",
        values="mean_accuracy"
    )

    print(f"\n===== Pivot (eps={eps_value}) =====")
    print(pivot)
def pivot_with_std(df, eps_value):
    sub = df[df["eps"] == eps_value]

    pivot_mean = sub.pivot_table(index="steps", columns="q", values="mean_accuracy")
    pivot_std = sub.pivot_table(index="steps", columns="q", values="std_accuracy")

    print(f"\n===== MEAN (eps={eps_value}) =====")
    print(pivot_mean)

    print(f"\n===== STD (eps={eps_value}) =====")
    print(pivot_std)
def main():
    # ===== 只筛选 multiseed 文件 =====
    logs = glob.glob("output/multiseed*.log")

    results = []
    for log_file in logs:
        try:
            results.append(parse_log(log_file))
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")

    df = pd.DataFrame(results)

    # ===== 去掉解析失败的 =====
    df = df.dropna(subset=["mean_accuracy"])

    # ===== 排序 =====
    df_sorted = df.sort_values(
    by=["mean_accuracy", "std_accuracy"],
    ascending=[False, True]
)

    # ===== 输出 =====
    print("\n===== ALL RESULTS (sorted by accuracy) =====\n")
    print(df_sorted.to_string(index=False))

    print("\n===== TOP 10 =====\n")
    print(df_sorted.head(10).to_string(index=False))

    # ===== 保存 CSV（强烈推荐）=====
    df_sorted.to_csv("output/summary.csv", index=False)
    print("\nSaved to output/summary.csv")
    analyze_fixed_eps(df)
    analyze_fixed_q(df)
    analyze_fixed_steps(df)
    # epslist = [0.01,0.05,0.1]
    # for eps in epslist:
    #       pivot_eps(df, eps)
    # for eps in epslist:
    #       pivot_with_std(df, eps)

if __name__ == "__main__":
    main()