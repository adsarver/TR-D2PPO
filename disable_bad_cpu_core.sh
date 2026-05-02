#!/usr/bin/env bash
set -euo pipefail

BAD_CPUS=(8 9)
SERVICE_PATH=/etc/systemd/system/disable-bad-cpu-core.service
SCRIPT_PATH=/usr/local/sbin/disable-bad-cpu-core

echo "Disabling suspect logical CPUs: ${BAD_CPUS[*]}"

for cpu in "${BAD_CPUS[@]}"; do
  cpu_online="/sys/devices/system/cpu/cpu${cpu}/online"
  if [[ -w "$cpu_online" ]]; then
    echo 0 > "$cpu_online"
  else
    echo "Cannot write $cpu_online; are you running with sudo?" >&2
    exit 1
  fi
done

cat > "$SCRIPT_PATH" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
for cpu in 8 9; do
  cpu_online="/sys/devices/system/cpu/cpu${cpu}/online"
  if [[ -e "$cpu_online" ]]; then
    echo 0 > "$cpu_online"
  fi
done
EOS
chmod 0755 "$SCRIPT_PATH"

cat > "$SERVICE_PATH" <<'EOS'
[Unit]
Description=Disable unstable CPU core 16 logical CPUs 8 and 9
After=sysinit.target
Before=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/disable-bad-cpu-core
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOS

systemctl daemon-reload
systemctl enable --now disable-bad-cpu-core.service

echo "--- current CPU state ---"
echo "online:  $(cat /sys/devices/system/cpu/online)"
echo "offline: $(cat /sys/devices/system/cpu/offline)"
echo "--- threads still on CPUs 8/9 ---"
ps -eLo pid,psr,comm,args | awk '$2==8 || $2==9 {print}' | head -20 || true

echo "Done. CPUs 8 and 9 should now be offline and will be disabled again at boot."
