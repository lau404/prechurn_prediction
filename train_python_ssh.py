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
    transport.connect(username='user_00', pkey=private_key)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh._transport = transport

    print(f"[SSH] 正在连接 {ip} 并执行命令：{cmd}", flush=True)
    stdin, stdout, stderr = ssh.exec_command(cmd)

    # 标准输出全收集
    stdout_lines = stdout.readlines()
    if stdout_lines:
        print(f"[远程标准输出] 共 {len(stdout_lines)} 行", flush=True)
        for line in stdout_lines:
            print(f"[STDOUT] {line.strip()}", flush=True)

    # 错误输出全收集
    err = stderr.read().decode()
    if err:
        print(f"[远程错误输出]\n{err}", flush=True)

    # 获取远程命令的退出码
    exit_code = stdout.channel.recv_exit_status()
    print(f"[SSH] 执行完成，返回码: {exit_code}", flush=True)

    ssh.close()

    # 若出错则抛出异常，让调度系统捕捉并记录日志
    if exit_code != 0:
        raise RuntimeError(f"[SSH错误] 远程命令失败，返回码: {exit_code}\n错误信息:\n{err}")




if __name__ == '__main__':
    # 1. 接收日期参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_date", required=True, help="调度运行日期，例如 20250302")
    args = parser.parse_args()
    run_date = args.run_date
    cmd = f"~/data/myenv/bin/python /home/user_00/data/temperedheroes/prechurn_script/train_python_hw.py --run_date {run_date}"
    print(cmd)
    ip = "192.168.1.30"
    ssh_execute(ip, cmd)
    print("--------------训练完成！----------------")
