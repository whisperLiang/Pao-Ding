if [[ $# -lt 1 ]]; then
  echo '请输入带宽上限，单位MB/s'
  exit
fi
# 在net_cls下创建名为mylim的cgroup
cd /sys/fs/cgroup/net_cls/ || exit
sudo su <<EOF
mkdir mylim
cd mylim/ || exit
echo 0x10010 > net_cls.classid
# 创建对于wlan0网卡的root规则，qdisc编号为1
tc qdisc add dev wlan0 root handle 1: htb
# 添加过滤器，使用cgroup号作为class
tc filter add dev wlan0 parent 1: handle 1: cgroup
# 添加class，对cgroup号为1:10的进程进行网速限制，单位为MB/s
tc class add dev wlan0 parent 1: classid 1:10 htb rate "$1""mbps"
echo 'rule created, all rules:'
tc class ls dev wlan0
EOF