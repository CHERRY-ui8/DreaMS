#!/bin/bash
# 统一使用 dreams conda 环境运行
exec conda run -n dreams "$@"
