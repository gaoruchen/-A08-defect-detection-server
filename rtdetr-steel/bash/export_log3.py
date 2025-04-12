import os

def extract_rtdetr_summary_from_file(log_file_path, max_lines=45):
    """
    从日志文件中提取以 'rtdetr-' 开头的行。
    :param log_file_path: 日志文件路径
    :param max_lines: 最大搜索行数，默认为 45
    :return: 提取的行（字符串），如果未找到则返回 None
    """
    with open(log_file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith('rtdetr-'):
                return line.strip()  # 去除行尾的空格
            if i >= max_lines - 1:
                break
    return None

def extract_last_n_lines(log_file_path, n=8):
    """
    从日志文件中提取倒数 n 行，保留每行前面的空格，去除后面的空格。
    :param log_file_path: 日志文件路径
    :param n: 需要提取的行数，默认为 8
    :return: 包含倒数 n 行的列表
    """
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            last_n_lines = lines[-n:]
            # 保留每行前面的空格，去除后面的空格和换行符
            last_n_lines = [line.rstrip() for line in last_n_lines]
            # 删除包含进度条的行
            last_n_lines = [line for line in last_n_lines if not line.startswith("100%|")]
            return last_n_lines
    except (FileNotFoundError, IOError):
        return ["Error reading file or file not found."]
    
def extract_map_values(data):
    """
    从输入的字符串列表中提取 all 行的 mAP50 和 mAP50-95 值。

    参数:
        data (list): 包含检测结果的字符串列表。

    返回:
        tuple: 包含 mAP50 和 mAP50-95 的元组，如果未找到则返回 (None, None)。
    """
    for line in data:
        if line.strip().startswith("all"):
            parts = line.split()
            map50 = float(parts[-2])  # 提取 mAP50
            map50_95 = float(parts[-1])  # 提取 mAP50-95
            return map50, map50_95
    return None, None  # 如果未找到 all 行，返回 None

def process_log_directory(directory_path, output_file, min_size_mb=60):
    """
    处理目录下的所有大于指定大小的 .log 文件，提取信息并输出到 combine.txt。
    :param directory_path: 日志文件目录路径
    :param output_file: 输出文件路径
    :param min_size_mb: 文件大小的最小值（MB），默认为 60MB
    """
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(directory_path):
            if filename.endswith(".log"):
                log_file_path = os.path.join(directory_path, filename)
                file_size_mb = os.path.getsize(log_file_path) / (1024 * 1024)  # 转换为 MB

                # 只处理大于指定大小的文件
                if file_size_mb > min_size_mb:
                    print(f"Processing file: {filename} (Size: {file_size_mb:.2f} MB)")
                    outfile.write(f"Processing file: {filename} (Size: {file_size_mb:.2f} MB)\n")


                    # 提取最后 8 行
                    last_8_lines = extract_last_n_lines(log_file_path)
                    map50, map50_95 = extract_map_values(last_8_lines)

                    outfile.write(f"{filename[:-4]}-{map50}-{map50_95}\n")
                    # 提取summary_line 信息
                    summary_line = extract_rtdetr_summary_from_file(log_file_path)

                   
                    

                    if summary_line:
                        outfile.write(f"{summary_line}\n")  # 直接输出提取的行，不加额外文本

                    
                    # 假设 last_8_lines 是你的输入数据
                    for i, line in enumerate(last_8_lines):  # 使用 enumerate 获取行号和行内容
                        if line.strip().startswith("Class"):
                            index = line.find(':')  # 查找关键字的位置
                            if index != -1:  # 如果找到关键字
                                line = line[:index]  # 删除冒号及之后的内容
                        # 只对前 6 行进行 line = line[15:] 操作
                        if i < 6:  # 前 6 行的索引是 0, 1, 2, 3, 4, 5
                            line = line[15:]  # 去除前 15 个字符
                        outfile.write(line + "\n")  # 写入文件

                    # 表格数据
                    parameter = int(summary_line.split(' ')[4])/1000000
                    parameter = round(parameter,1)
                    glops = summary_line.split(' ')[8]
                    outfile.write("-" * 40 + "\n")
                    outfile.write(f"{filename[:-4]} {parameter} {glops} {map50} {map50_95}\n")

                    # 在每个日志文件之间加入 10 行空行
                    outfile.write("\n" * 10)
                else:
                    print(f"Skipping file: {filename} (Size: {file_size_mb:.2f} MB)")

# 示例使用
log_directory = "logs"  # 替换为你的日志目录路径
output_file = "logs/combine.txt"  # 输出文件路径
process_log_directory(log_directory, output_file)