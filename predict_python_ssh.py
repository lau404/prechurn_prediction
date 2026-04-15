import argparse
import copy
import hashlib
import json
import os
import sys

import pymysql
import paramiko
import subprocess


# 默认用户user_00
def ssh_start_execute(ssh_client, cmd):
    stdin, stdout, stderr = ssh_client.exec_command(cmd, timeout=60)
    return stdin, stdout, stderr

def ssh_execute(ip, cmd):
    private_key = paramiko.RSAKey.from_private_key_file('/home/user_00/.ssh/id_rsa')
    transport = paramiko.Transport((ip, 22))
    transport.connect(username='***', pkey=***)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 允许首次连接
    ssh._transport = transport

    print(f"[SSH] 正在连接 {ip} 并执行命令：{cmd}", flush=True)
    stdin, stdout, stderr = ssh.exec_command(cmd)

    # 读取标准输出
    for line in iter(stdout.readline, ""):
        print(f"[远程输出] {line.strip()}", flush=True)

    # 读取错误输出
    err = stderr.read().decode()
    if err:
        print(f"[远程错误] {err.strip()}", flush=True)

    exit_code = stdout.channel.recv_exit_status()
    print(f"[SSH] 执行完成，返回码: {exit_code}", flush=True)

    ssh.close()
    if exit_code != 0:
        print("-----------------------")
        sys.exit(-1)



if __name__ == '__main__':
    # 1. 接收日期参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_date", required=True, help="调度运行日期，例如 20250302")
    args = parser.parse_args()
    run_date = args.run_date
    cmd = f"~***/predict_python_hw.py --run_date {run_date}"
    print(cmd)
    ip = ***
    ssh_execute(ip, cmd)
    print("--------------训练完成！----------------")
