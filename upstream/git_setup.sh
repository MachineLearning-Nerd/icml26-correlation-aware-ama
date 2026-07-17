#!/bin/bash

# 使用说明:
# 1. 修改下面的 REMOTE_URL 为你的仓库地址
# 2. 在终端运行: chmod +x git_setup.sh && ./git_setup.sh

REMOTE_URL="https://github.com/yourusername/your-repo-name.git"

echo "=== 初始化 Git 仓库 ==="
git init

echo "=== 添加远程仓库 ==="
git remote add origin $REMOTE_URL

echo "=== 添加所有文件 ==="
git add .

echo "=== 创建初始提交 ==="
git commit -m "Initial commit: CA-AMA implementation for ICML 2026"

echo "=== 推送到远程仓库 ==="
git branch -M main
git push -u origin main

echo "=== 完成! ==="
