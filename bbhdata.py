import glob
import re
from typing import List, Tuple, Dict

def extract_perf_data(file_pattern: str = "output/bbh-*") -> Dict[str, List[Tuple[float, float]]]:
    """
    提取匹配文件模式的所有文件中的性能数据。

    参数:
        file_pattern: 文件匹配模式，默认 "output/bbh-*" 匹配 output 文件夹下所有以 bbh- 开头的文件

    返回:
        字典，键为文件名（包含路径），值为按顺序提取的 (average_perf, best_perf) 元组列表
    """
    results = {}

    for file_path in glob.glob(file_pattern):
        perf_list = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # 匹配模式：average perf: 51.282051282051285 best perf: 57.692307692307686
                    match = re.search(
                        r' average\s+perf:\s*([\d.]+)\s+best\s+perf:\s*([\d.]+) ',
                        line,
                        re.IGNORECASE
                    )
                    if match:
                        avg_val = float(match.group(1))
                        best_val = float(match.group(2))
                        perf_list.append((avg_val, best_val))
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            continue

        if perf_list:
            results[file_path] = perf_list

    return results

def print_results(results: Dict[str, List[Tuple[float, float]]]) -> None:
    """友好地打印提取结果。"""
    if not results:
        print("未找到任何匹配的数据。请确认 output 文件夹中存在 bbh-* 文件。")
        return

    for file_name, data_list in results.items():
        print(f"\n文件: {file_name}")
        print(f"共提取 {len(data_list)} 条记录:")
        for idx, (avg, best) in enumerate(data_list, 1):
            print(f"  {idx:3d}. average perf = {avg:10.6f}, best perf = {best:10.6f}")

if __name__ == "__main__":
    # 提取 output 文件夹下所有 bbh- 开头的文件中的数据
    perf_data = extract_perf_data("output/bbh*")
    print_results(perf_data)

    # 如果需要将结果保存为 CSV 文件，取消以下注释
    # import csv
    # with open('extracted_perf_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['file', 'index', 'average_perf', 'best_perf'])
    #     for file, records in perf_data.items():
    #         for idx, (avg, best) in enumerate(records, 1):
    #             writer.writerow([file, idx, avg, best])