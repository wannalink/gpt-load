#!/bin/sh
set -e

# Setup and enable 512MB swap on host if not already active
if [ -f /proc/swaps ] && [ "$(wc -l < /proc/swaps)" -le 1 ]; then
    if [ ! -f /swapfile ]; then
        fallocate -l 512M /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=512 2>/dev/null
        chmod 0600 /swapfile
        mkswap /swapfile >/dev/null 2>&1
    fi
    swapon /swapfile 2>/dev/null || true
    echo 10 > /proc/sys/vm/swappiness 2>/dev/null || true
    echo 1 > /proc/sys/vm/overcommit_memory 2>/dev/null || true
fi

# Step down to non-root user 10001 (gpt-load) and execute binary
if [ "$(id -u)" -eq 0 ]; then
    chown -R 10001:10001 /app/data 2>/dev/null || true
    exec su-exec 10001:10001 /app/gpt-load "$@"
else
    exec /app/gpt-load "$@"
fi
