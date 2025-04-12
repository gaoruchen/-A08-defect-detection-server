import os
import re

def format_number_with_unit(number_str):
    number = int(number_str)
    units = ["", "K", "M", "G", "T", "P"]
    unit_index = 0

    while number >= 1000 and unit_index < len(units) - 1:
        number /= 1000.0
        unit_index += 1

    return f"{number:.1f}{units[unit_index]}"
def extract_and_format_log_info_from_all_logs(log_dir_path, export_path, max_lines=500, num_last_lines=8):
    summary_pattern = re.compile(r"(\S+) summary: (\d+) layers, (\d+) parameters, \d+ gradients, [\d.]+ GFLOPs")
    all_logs_content = []
    # 将所有日志内容写入到一个txt文件中，每个日志内容间隔10行
    with open(export_path, 'w') as export_file:
        with open(result_path,'w') as result_file:
            result_file.write('name,parameters,mp50,mp50-95\n')
            for log_filename in os.listdir(log_dir_path):
                if log_filename.endswith('.log'):
                    log_file_path = os.path.join(log_dir_path, log_filename)
                    summary_line = None
                    parameters = "N/A"
                    model_name = "N/A"
                    last_lines = []

                    with open(log_file_path, 'r', encoding='utf-8') as log_file:
                        lines = log_file.readlines()
                        # 提取前500行中的summary信息
                        for i, line in enumerate(lines[:max_lines]):
                            match = summary_pattern.search(line)
                            if match:
                                summary_line = line.strip()
                                model_name = match.group(1)  # 捕获名字信息
                                parameters = match.group(3)  # 捕获参数信息
                                break

                        # 提取最后8行
                        last_lines = lines[-num_last_lines:]

                    tmp = []
                    for line in last_lines:
                        line = line.lstrip().strip()
                        tmp.append(line)
                    last_line = tmp
                    yaml = summary_line.split()[0] + '.yaml'
                    # 将提取到的内容添加到all_logs_content中
                    if summary_line:
                        export_file.writelines(f"{summary_line}\n")
                    else:
                        print("Summary information not found in the first 500 lines of the log file.")
                        export_file.writelines("Summary information not found in the first 500 lines of the log file.\n")

                    mp50 = 0
                    person_mp50 = 0
                    mp5095 = 0
                    # 格式化输出2-7行
                    try:
                        for line in last_lines[:6]:
                            line = line.lstrip()
                            parts = line.split()
                            try:
                                if parts[0] == 'all':
                                    mp50 = parts[5]
                                    mp5095 = parts[6]
                                if parts[0] == 'person':
                                    person_mp50 = parts[5]
                            except:
                                print("An exception occurred")
                            formatted_line = "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(*parts)
                            export_file.write(f"{formatted_line}\n")
                    except:
                        for line in last_lines[:6]:
                            export_file.write(f"{line}\n")
                    # 输出剩余的行，去除前面的空格
                    for line in last_lines[6:]:
                        export_file.write(line.strip() + "\n")

                    formatted_line = "{}-{}-{:>4}-{:>4}-{:>4}".format(yaml, format_number_with_unit(parameters),mp50, person_mp50, mp5095)
                    formatted_line2 = "{},{},{},{}".format(model_name, format_number_with_unit(parameters), mp50, mp5095)
                    export_file.write(f"{formatted_line}\n")
                    result_file.write(f"{formatted_line2}\n")
                export_file.write("\n" * 10)


# 示例用法
log_dir_path = r'/home/lihan/Code/rtdetr/logs'
export_path = r'/home/lihan/Code/rtdetr/logs/combined_logs.txt'
result_path = r'/home/lihan/Code/rtdetr/logs/result.txt'

# 提取和格式化日志信息，并保存到combined_logs.txt
extract_and_format_log_info_from_all_logs(log_dir_path, export_path)
