# Pao-Ding
Pao-Ding: Automatic DNN Parsing and Decomposition for Collaborative Inference

## 环境配置

Python版本3.10.12。

### Ubuntu&树莓派&Win

安装Python依赖包
```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 网络配置

### 拓扑结构

要确保如下的设备访问路径可行，即client可以通过特定的IP端口号访问到相应的server。

| 设备(client) | 要访问的设备(server) |
| ------------ | -------------------- |
| Master       | 所有Worker，Trainer  |
| Worker i     | Worker i+1           |

虽然Worker也会作为client请求Master，但是系统内部通过grpc的流式回复实现了这一功能，所以无需配置。

### 网络限速

使用 [cgroup](https://man7.org/linux/man-pages/man7/cgroups.7.html) 和 [tc](https://man7.org/linux/man-pages/man8/tc.8.html) 对指定进程进行限速。基本原理就是，tc对指定cgroup的所有进程限制总带宽为一个特定值。这里我们只需要对一个进程限速，所以cgroup中只会有一个进程。

#### 安装

tc系统自带，只需安装cgroup工具，以便使用cgexec命令。

```bash
sudo apt install cgroup-tools
```

#### 创建cgroup和tc规则

##### 创建命令

对网卡wlan0创建带宽上限为1MB/s的cgroup命令如下：

```bash
# 在net_cls下创建名为mylim的cgroup
cd /sys/fs/cgroup/net_cls/
sudo su
mkdir mylim
cd mylim/
echo 0x10010 > net_cls.classid
# 创建对于wlan0网卡的root规则，qdisc编号为1
tc qdisc add dev wlan0 root handle 1: htb
# 添加过滤器，使用cgroup号作为class
tc filter add dev wlan0 parent 1: handle 1: cgroup
# 添加class，cgroup号为1:10的进程网速限制为1MB/s
tc class add dev wlan0 parent 1: classid 1:10 htb rate 1mbps
```

上述命令已封装在tool/set_net_lim.sh里，使用方法：

```bash
bash tool/set_net_lim.sh <带宽上限, 单位MB/s>
```

如`bash tool/set_net_lim.sh 4`表示创建一个名为mylim的cgroup，这个cgroup下面的进程带宽上限为4MB/s.

##### 开机自动配置

cgroup配置关机以后会失效。

因为树莓派经常开关机，所以可以把上述的cgroup配置脚本配置到开机启动项中。

以带宽上限为4MB/s为例，在/etc/rc.local文件的`exit 0`前添加如下代码，相应的log文件会记录配置后的所有tc class。

```bash
bash /home/pi/cnn-video/tool/set_net_lim.sh 4 > /home/pi/cnn-video/tool/set_net_lim.log
```

##### 相关命令

查看设备wlan0上已有的规则（输出的Mbit表示Mbit/s，除以8为MB/s）

```bash
tc class ls dev wlan0
```

修改规则中的限速为4MB/s（把上面添加命令中的add改成change，1mbps改成4mbps）

```bash
sudo tc class change dev wlan0 parent 1: classid 1:10 htb rate 4mbps
```

#### 把服务运行在指定cgroup中

以启动worker0为例，命令如下（即lim_net_start.sh）

```bash
sudo -E PATH=$PATH PYTHONPATH=$(python -c "import sys; print(':'.join(sys.path))") cgexec -g net_cls:mylim /usr/bin/python3.7 main.py worker -i 0
```

`cgexec -g net_cls:mylim` 指定后面的命令启动的进程运行在net_cls下的mylim这个cgroup中

cgexec需要sudo权限，但是因为Python的包都是安装在当前用户下而不是系统安装，所以root用户目录下运行python会找不到很多包。所以这里要设置PATH和PYTHONPATH（就是`sys.path`）告诉解释器从哪里加载包。此外，sudo的E选项表示运行后面命令时保留当前的环境变量。

## CPU限制

同样使用cgroup进行限制。比如限制mylim下的进程CPU占用率为40%，即只能使用一个核的40%（8个核总共可以用800%），使用如下代码

```bash
cd /sys/fs/cgroup/cpu,cpuacct
sudo mkdir mylim
cd mylim
# quota为时间周期长度内能使用的CPU时间，单位微秒(us)
echo 100000 > cpu.cfs_quota_us
# period为时间周期长度，单位微秒(us)
echo 250000 > cpu.cfs_period_us
```

类似地，在该cgroup中启动，命令如下（即lim_cpu_start.sh）

```bash
sudo -E PATH="$PATH" PYTHONPATH=$(python -c "import sys; print(':'.join(sys.path))") cgexec -g cpu,cpuacct:mylim /usr/bin/python3 main.py "$@"
```

## 运行实验

网络就绪后，就可以运行实验了。

首先在rpc目录下运行脚本msg_compile.cmd，生成grpc文件。

然后在config.yml中填写相关配置。

启动Trainer

```bash
python3 main.py trainer
```

如果不限速直接启动各个Worker，如Worker0：

```bash
python3 main.py worker -i 0
```

如果已经按照之前所述的方法配置了cgroup和tc规则，并且要限制Worker0的出站速度，则通过lim_start.sh进行启动，命令如下：

```bash
/bin/bash lim_start.sh worker -i 0
```

最后启动Master（注意Master要在Worker和Trainer之后启动）

```bash
python3 main.py master
```

类似地，要限速的话，命令是：

```bash
/bin/bash lim_start.sh master
```

> **小tip**：如果想把执行lim_start.sh配置到PyCharm的Run/Debug Configuration中的话，可以添加一个SSH的Python解释器，但是这个Python解释器的路径设置成/bin/bash。其他的都和远程执行Python脚本一样。
