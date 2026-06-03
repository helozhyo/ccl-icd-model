#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSH helper for autodl server.
"""
import paramiko
import sys

HOST = 'YOUR_SERVER_HOST'
PORT = 00000  # your SSH port
USER = 'root'
PASSWORD = 'YOUR_PASSWORD'


def ssh_exec(cmd, timeout=60):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    try:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        return out, err
    finally:
        client.close()


if __name__ == '__main__':
    cmd = ' '.join(sys.argv[1:])
    out, err = ssh_exec(cmd)
    print(out)
    if err:
        print('STDERR:', err)
