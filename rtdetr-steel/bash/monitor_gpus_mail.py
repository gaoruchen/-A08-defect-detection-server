import smtplib
import time
import socket
import subprocess
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

# 从环境变量中获取敏感信息
username = os.getenv('QQ_EMAIL_USERNAME', '253514942@qq.com')  # QQ邮箱用户名
password = os.getenv('QQ_EMAIL_PASSWORD', 'tpfzerwwjizcbgdf')  # QQ邮箱授权码

# 发送地址和接收地址
send_mail = '253514942@qq.com'
receive_mail = 'lihan3@qq.com'

# 邮件服务器配置
mail_host = 'smtp.qq.com'
mail_port = 587

# 显卡数量
num_gpus = 4

# 显存状态阈值（单位：MiB）
MEMORY_THRESHOLDS = [5120, 10240, 15360, 20480]  # 5G, 10G, 15G, 20G

# 初始化显存状态
gpu_memory_states = [None] * num_gpus  # 用于存储每张卡的显存状态

def get_gpu_memory():
    """
    使用 nvidia-smi 命令获取每张 GPU 的空闲显存（单位：MiB）。
    """
    try:
        # 调用 nvidia-smi 命令
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        # 提取每张 GPU 的空闲显存
        memory_list = [int(x) for x in output.strip().split('\n')]
        return memory_list
    except Exception as e:
        print(f"获取 GPU 显存失败: {e}")
        return [0] * num_gpus  # 返回默认值

def get_memory_state(free_memory):
    """
    根据空闲显存返回状态。
    """
    if free_memory >= MEMORY_THRESHOLDS[3]:
        return ">20G"
    elif free_memory >= MEMORY_THRESHOLDS[2]:
        return ">15G"
    elif free_memory >= MEMORY_THRESHOLDS[1]:
        return ">10G"
    elif free_memory >= MEMORY_THRESHOLDS[0]:
        return ">5G"
    else:
        return "<5G"

def send_email(send_mail, receive_mail, subject, body):
    """
    发送邮件。
    """
    # 创建邮件对象
    msg = MIMEMultipart()
    msg['From'] = send_mail
    msg['To'] = receive_mail
    msg['Subject'] = subject

    # 添加邮件正文
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    # 发送邮件
    try:
        server = smtplib.SMTP(mail_host, mail_port)
        server.set_debuglevel(1)
        server.ehlo()
        server.starttls()
        server.login(username, password)
        server.sendmail(send_mail, receive_mail, msg.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(f"邮件发送失败: {e}")
    finally:
        server.quit()

def monitor_gpus():
    """
    监控 GPU 显存状态，并在状态变化时发送邮件。
    """
    global gpu_memory_states
    
    # 获取并发送初始状态
    current_memory = get_gpu_memory()
    current_states = [get_memory_state(mem) for mem in current_memory]
    gpu_memory_states = current_states

    # 发送初始状态邮件
    subject = "GPU 显存初始状态"
    body = "当前 GPU 显存状态：\n"
    for i in range(num_gpus):
        body += f"GPU {i}: {current_memory[i]} MiB ({current_states[i]})\n"
    send_email(send_mail, receive_mail, subject, body)
    print("已发送初始状态邮件")

    # 开始监控循环
    while True:
        current_memory = get_gpu_memory()
        current_states = [get_memory_state(mem) for mem in current_memory]

        if current_states != gpu_memory_states:
            print("显存状态发生变化，发送邮件通知...")
            gpu_memory_states = current_states

            subject = "GPU 显存状态更新"
            body = "当前 GPU 显存状态：\n"
            for i in range(num_gpus):
                body += f"GPU {i}: {current_memory[i]} MiB ({current_states[i]})\n"

            send_email(send_mail, receive_mail, subject, body)
        else:
            print("显存状态未发生变化")

        time.sleep(600)

if __name__ == "__main__":
    monitor_gpus()