#!/bin/bash
echo "操作系统："
grep PRETTY_NAME /etc/os-release | cut -d'"' -f2
echo ""
echo "CPU："
lscpu | grep "Model name" | sed 's/Model name: *//'
echo "核心数：$(lscpu | grep '^Core(s) per socket' | awk '{print $4}') 核，线程数：$(nproc)"
echo ""
echo "内存："
free -h | awk '/^Mem:/{print $2}'
echo ""
echo "GPU："
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "无GPU"
echo ""
echo "系统盘："
lsblk -d -o NAME,SIZE,MODEL | grep -v "loop" | head -2
df -h / | awk 'NR==2{print $2}'