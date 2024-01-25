# 开发笔记

## 2022.2.17

- [x] [实验] 在vtrace中添加了传输总耗时total_transmit的统计

## 2022.1.29

- [x] [实验] 添加了gp_show，以展示不同gp_size下的总耗时，作为论文图表
- [x] [实验] 在latency_show中添加了vg16的结果图
  * 发现vg16的效果较差，暂且不放入论文中

## 2022.1.28

- [x] [实验] vtrace代码整理，使用3种模式读取tc文件。3种模式测试都正常
- [x] IFRTracker添加了对前n帧时延均值的统计。协同测试运行正常
- [x] [实验] vtrace添加了XLIM选项，以便生成一致的图
- [x] [实验] 添加了latency_show，以展示不同队列长度下的单帧时延和时延均值，作为论文图表

## 2022.1.27

- [x] [实验] vtrace画图代码修改，以便生成论文里的图

## 2022.1.25

- [x] 添加了IFRTracker用于监控IFR状态，MyScheduler调度考虑了当前IFR状态，修复了MyScheduler没有更新缓存记录的问题。2个Worker协同测试正常。代码还没整理好。
  * IFRTracker：记录IFR的状态，向Scheduler提供当前IFR的状态以便调度
  * Master：一部分功能移动到了IFRTracker中
  * MasterServicer：把Master放到了主线程中。Master不需要继承Thread了
  * Scheduler：LBScheduler.elys2olys移动到了这里
  * msg.proto：Worker添加了上报stage状态的rpc和相关数据结构
  * stub_factory：修改相关类以配合新增的finish_stage
  * LBScheduler：elys2olys移动到了Scheduler中
  * Metric：添加了阶段完成时间s_ready参数，整理了代码，simulate_pipeline以阶段为单位进行模拟，不再区分传输和计算
  * MyScheduler：调度时考虑了现有的IFR状态，修复了没有更新`pre_wk_ilys`的bug。gen_ifr_group的参数还没统一
  * Worker：在完成传输和计算时向Master上报
  * WorkerServicer：添加了相关数据结构以配合新增的finish_stage
  * 目前在2个Worker上测试的配置上表现都不差于之前的MyScheduler（如果差，重跑一下可能就好了），也都好于LBScheduler。测试的配置包括：
    * ax.road: ifr_num=10, pd_num=0, gp_size=3
    * vg16.road: ifr_num=10, pd_num=0, gp_size=3
    * ax.parking: ifr_num=10, pd_num=0, gp_size=3
    * vg16.parking: ifr_num=12, pd_num=3, gp_size=3

- [x] 整理了代码，调整了IFRTracker和Scheduler的交互方式，Master不再是Thread，config顺序调整。2个Worker协同测试正常

## 2022.1.22

- [x] [实验] 修改了lcnz_show，以生成论文里的图
  * 生成的图为cnz-org-ax.road.480x720.400.1.pdf和cnz-dif-ax.road.480x720.400.1.pdf，org和dif分别表示原始数据和差值数据，末尾的1表示`TARGET_FRAME=1`
- [x] [实验] 修改了lsz_show，以统计论文所需数据
- [x] 整理了MyScheduler的代码，抽象出了Metric，添加了recur_find_chain用于处理多个Worker。协同测试正常
- [x] MyScheduler支持了多个Worker。2个Worker协同测试正常
- [x] MyScheduler小bug修复，使得可以正常运行3个Worker。3个Worker协同测试正常
  * 3个Worker的实验配置：pi4G(m) → 4MB/s(实际带宽3MB/s) → pi2G(w0) → 4MB/s(实际带宽3MB/s) → pi2G1(w1) → 4MB/s(实际带宽3MB/s) → pi2G2(w2). 标注的4MB/s：tc限速4MB/s，实际包括了编解码以后带宽会更少，3MB/s为log输出的估计带宽
  * 3个Worker实验发现，MyScheduler性能差于LBScheduler
  * 目前pi2G1配置了网络限制的cgroup

- [x] [实验配置] 在parking.mp4上进行了测试。2个Worker（pi2G+pi2G1）协同测试正常

## 2022.1.21

- [x] 优化了线性(LNR)和多个MLP(MLPs)predictor的预测耗时，通过lsz_show验证了效果。协同测试正常
- [x] [实验] 添加了cps_perf_show，为陈智麒帮忙画的压缩方法性能对比图
- [x] [实验] 修改了cps_perf_show，以便画论文里的图
- [x] [实验] 修改了lsz_show，以生成论文里的图

## 2022.1.20

- [x] [文档] 添加了对开机自动配置带宽限制cgroup的说明
  * 目前pi4G和pi2G已经配置，运行正常

- [x] LBCScheduler可以处理DAG了，更名为LBScheduler。协同测试正常

## 2022.1.19

- [x] Master中添加了各IFR的时延记录，MyScheduler候选点添加了len(dag)，添加了设置带宽限制的脚本。协同测试正常
- [x] RawDNN生成的HTML名称为CNN名称，不再是dag_layers.html。chain测试正常

## 2022.1.14

- [x] [文档+实验] 添加了限制CPU使用率的方法和相关脚本，实验设备从pi2G1+aliyun改成了pi2G+pi2G1。协同测试正常
  * 修改实验设备的主要原因是，aliyun性能超过pi2G1太多了，会导致MyScheduler对于vg16, gn, rs50都会全部调度到云上执行
  * 新的实验配置：pi4G(m) → 4MB/s(实际带宽3MB/s) → pi2G(w0) → 4MB/s(实际带宽3MB/s) → pi2G1(w1). 标注的4MB/s：tc限速4MB/s，实际包括了编解码以后带宽会更少，3MB/s为log输出的估计带宽
  * 不使用pi3Bp的原因：pi3Bp只有1G的内存，在profile时会崩溃，因此pi3B, pi3Bp, pi1G都不能使用
  * 之后可能使用的设备：pi2G, pi2G1, pi2G2, pi2G3, pi4G

## 2022.1.13

- [x] [实验] lnz_show添加了对原始数据非零率的展示，数据集从cache中加载
- [x] [实验] 新增了lsz_show展示原始数据(不压缩/压缩)、差值压缩数据(实际值/预测值)的数据量。运行正常
- [x] MyScheduler通过找主干点的方式实现了对DAG的支持，lsz_show添加了对主干节点的标识。云边协同正常，lsz_show运行正常
  * 观察发现，VGG16确实是输入数据的数据量很小，导致把它放在云端是最合适的

- [x] 修改了GRPC设置，解决了RPC消息过大的问题。云边协同正常

## 2022.1.11

- [x] [实验] ionz_fit添加了不显示拟合曲线的参数。运行正常
- [x] [画图] ionz_fit生成了毕业论文中所用的图：ax.road.480x720.400-cv.eps，ax.road.480x720.400-rl.eps，ax.road.480x720.400-mp.eps
- [x] 系统中使用原始数据非零占比O_LFCNZ的均值对原始数据大小进行估计。本地协同和云边协同均正常

## 2022.1.10

- [x] 优化了Master输出，vtrace默认为远程模式。云边测试正常
- [x] [实验] lfcnz_gen添加了原始数据对应的稀疏表示格式OLFCNZ。运行正常
  * 实验发现，原始数据本身稀疏程度就很高（有50%），差值只是让前面几层的稀疏率提升了

## 2022.1.9

- [x] 整理了Scheduler的API，使之支持以group为单位的调度，相应修改了现有的调度器。所有调度器本地测试均运行正常
- [x] 解决了Ctrl-C不能直接退出的问题。本地测试和云边协同测试均正常
  * Linux平台上都是直接点一下就可以退出了，Windows平台上Master如果先退出的话可能有时候需要点两下。但是一般都是一起退出，此时还是可以直接退出的

- [x] vtrace添加了自动从远程下载tc文件的功能。本地测试和云边协同测试均正常

配置文件为device.yml，因为里面有IP地址和用户名密码，所以不纳入版本管理。格式如下：

```yaml
role:  # 这里写各个角色对应的设备名，如pi2G1, pi4G, aliyun
  m: pi2G1
  w: [pi4G, aliyun]

device:  # 各个设备名对应的登录用户和ssh地址
  pi2G1:
    user: pi  # 用户名
    addr: x.x.x.x:22  # SSH对应的IP地址和端口号
  # pi4G和aliyun也类似填写

user:  # 各个用户的登录密码
  pi: password  # 这里填用户对应的密码
```

## 2022.1.8

- [x] 修改了master，IFR以group的形式进行调度，添加了MyScheduler（Scheduler接口还没改）。EC协同运行有其他原因导致的问题，添加了相关TODO
- [x] master中最后一个IFR完成时对所有IFR是否完成进行检查。EC协同运行符合预期，即Master会报出没有正常完成的IFR
- [x] 去掉了master中没用的检查，MasterStub中使用AsyncClient确保同一个设备按序处理IFR，去掉了worker中的一致性检查，修复了vtrace中的bug。EC协同运行正常，check=true时运行正确
  * 去掉master中没用的检查：IFR可能会乱序完成，此时没必要报错
  * 修复vtrace中的bug：因为有的worker可能只完成了多个IFR中的几个，所以在读取worker的trace时要指定IFR总数量
  * 去掉worker中IFR id的一致性检查：假设有这样的执行计划，前3帧都是[1, 2] + []，第4帧是[] + [1, 2]。在第4帧时，w0会发送给w1的是第1层的原始数据，所以w1是可以正确执行得到结果的。但是一致性检查却会在此时报错，因为上一帧是-1。这显然不符合预期。要确保worker的正确性，主要是确保下一个worker要么接收的是完整数据，要么接收的是已缓存的差值数据。
    * w0总是从master接收差值数据，所以输出一定是正确的
    * 当w0的输出不在OutCache中时，便会传给w1完整数据，此时w1的输出一定是正确的
    * 当w0的输出在OutCache中时，便会传给w1差值数据。如果w0的输出在OutCache中，说明w0上次的输出层和这次一样。上次有两种可能，要么直接发给了w1，要么发给了master。前一种情况下，w1已经有了上次的缓存，便可以得到正确输出。后一种情况下，如果w0传给了master而没有发给w1，说明上一个IFR在w0就完成了，但是现在又要传给w1，说明现在w0并没有完成IFR，也就是说OutCache中只有最终的输出层而现在要传输的层完全是中间层，也就是说，此时OutCache已经完全失效了，那么此时传输的就是完整数据而非差值数据。参见上一种情况，w1的输出仍然正确
    * 依次类推，后面的worker输出也都是正确的
    * 总结：只要同一个IFR被worker处理的顺序固定不变，即使调度策略变化，已完成的worker直接跳过后续worker汇报给master，这些worker得到的输出都是正确的

- [x] master中修改了所有IFR都完成的检查逻辑。EC协同测试正常
- [x] 把Master中IFRGroup大小交给Scheduler决定，main中添加了config输出。EC协同测试正常

## 2022.1.6

- [x] :four_leaf_clover: [实验] 修改了lnz_show，以便为毕业论文生成AlexNet的相关图示

- [x] :four_leaf_clover: 添加了OSCScheduler作为baseline。云边协同测试正常

功利的角度，为什么需要在线调度：如果调度方案是离线的，那么就不需要非零率预测了。我可以直接用数据集里的平均值来生成方案。所以为了能让预测这件事情有用，我们需要做在线调度。

科研的角度，为什么需要在线调度：从AlexNet的热力图来看，在线调度的必要性主要是因为某些层差值数据非零率的波动很大，这就导致我们无法用离线的数据预知所有帧在这一层的非零率，也就无法预知传输数据量。而传输数据量会影响到我们的调度策略，所以我们需要在线地对各帧的非零率进行预测。

为了设计更好的调度器，首先观察一下baseline的表现：纯边缘和纯云端。

dif模式下，OSCScheduler[edge]，即全部都在边缘W0-pi4G上完成。3帧总耗时为5.3秒，平均1.8秒。

![image-20220107111426364](md-img/image-20220107111426364.png)

dif模式下，OSCScheduler[cloud]，即全部都在云端W1-aliyun上完成。3帧总耗时为7.3秒，平均2.4秒。

![image-20220107112351311](md-img/image-20220107112351311.png)

## 2022.1.5

- [x] 添加了对中间数据的抽象IMData以及相关派生类，相应地修改了faster_rcnn相关代码。frcnn代码运行正常
  * frcnn里面主要是粗略地展示了各层输出数据差值的稀疏性，目前看来，如果不把作为backbone的ResNet50拆开的话，只有原始数据和transform后的数据是稀疏的（非零占比10%以下），其他的非零占比都在99%以上。此外，最后的detections数据量很小，所以不考虑压缩（因此添加了CpsIM和UCpsIM）
  * 接下来计划把重心放在毕业论文上，即只处理链状的用于分类的CNN，不考虑目标检测CNN，所以暂时不使用IMData这个类

## 2021.12.29

- [x] 添加了Faster RCNN的代码，可以生成RawDNN，且RawDNN正常运行。frcnn运行正常
  - [x] raw_dnn.py：各层的输出数据不再是Tensor，而是Any
- [x] :four_leaf_clover: [实验] 用frcnn.py测试Faster RCNN跑在ItgExecutor上，运行正常

## 2021.12.28

- [x] :four_leaf_clover: [实验] 添加了Faster RCNN的测试代码，可以正常运行

## 2021.12.27

- [x] :four_leaf_clover: [实验] 使用tc+cgroup的方法对设备之间带宽限速成功。

在像昨天那样配置好cgroup和tc规则之后，启动服务时要用如下命令：

```bash
sudo -HE PATH=$PATH PYTHONPATH=$(python -c "import sys; print(':'.join(sys.path))") cgexec -g net_cls:mylim /usr/bin/python3.7 main.py worker -i 0
```

> cgexec不用解释，关键是sudo后面需要添加PATH和PYTHONPATH。因为sudo执行的python和普通用户的PYTHONPATH不一样，所以要手动注入当前环境的PYTHONPATH。

两个Worker，m->w0->w1用上述方法限速4MB/s和1MB/s。虽然WorkerStub的log显示只有3.1MB/s和0.9MB/s，但这是因为这里的时延包含了解码时间，所以实际上应该基本接近限速的耗时。

itg模式下LBCScheduler效果如下，可以看到Worker的各个IFR传输耗时均等，有限速效果。

![image-20211227094855448](md-img/image-20211227094855448.png)

- [x] [文档] 在README中添加了使用cgroup+tc限速的方法
- [x] :four_leaf_clover: [实验] 添加了lim_start.sh，在README中添加了用此脚本限速和PyCharm配置的方法。云边协同测试执行正常，限速正常。

## 2021.12.26

- [x] :four_leaf_clover: [实验] 两个Worker，m->w0->w1限速4MB/s和1MB/s。同样的LBCScheduler，对比了itg和dif模式下的效果。

itg模式

<img src="md-img/Snipaste_2021-12-26_11-30-23.png"  />

dif模式

![](md-img/Snipaste_2021-12-26_11-34-06.png)

从上图可以看到，第一帧的耗时相同，但是dif模式下后面两帧的耗时大大缩短了。

---

- [x] :four_leaf_clover: [实验] 添加了对trickle限速的实验结论

前面的限速是使用wondershaper对网卡进行限速，但是这样会影响其他进程的网速（如ssh），我现在希望能对这个进程进行限速。

[trickle](https://github.com/mariusae/trickle) 可以针对单个进程限速，但是效果不是太好，如下图就是itg模式下LBC调度器的结果。w0(pi4G)->w1(aliyun)每次发送的数据量相同，但是IFR2的耗时大大短于前面两个。改成10个IFR之后，就会变成IFR9的耗时明显短于前面的。trickle的用法参考 [Linux限制网络带宽的占用](https://developer.qiniu.com/evm/kb/2512/linux-network-bandwidth-utilization)。

![image-20211226152447349](md-img/image-20211226152447349.png)

---

- [x] [文档] 使用tc对指定进程的网络进行了限速，Python的http.server出站带宽限制成功

以pi4G（worker0）为例。因为tc的流量整形限制的都是出站规则，限制它到aliyun（worker1）之间的带宽为1MB/s。

```bash
# 创建cgroup
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

安装cgroup-tools：`sudo apt install cgroup-tools`

启动服务，将http.server运行在指定的cgroup中：

```bash
sudo cgexec -g net_cls:mylim python -m http.server 8000
```

其他设备从它这里下载时，网速就只有1MB/s了。

删除指定class

```bash
tc class delete dev wlan0 parent 1: classid 1:10
```

清除wlan0网卡上所有规则

```bash
tc qdisc delete dev wlan0 root
```

参考链接：

* [How can I limit Download bandwidth of an existing process? (iptables, tc, ?)](https://unix.stackexchange.com/questions/328308/how-can-i-limit-download-bandwidth-of-an-existing-process-iptables-tc)
* [[译] 《Linux 高级路由与流量控制手册（2012）》第九章：用 tc qdisc 管理 Linux 网络带宽](https://arthurchiao.art/blog/lartc-qdisc-zh/)
* [tc流量控制（一）](http://www.361way.com/tc-qos-1/1207.html)

## 2021.12.25

- [x] 修复了vtrace中最后传输事件丢失的问题。vtrace测试正常
- [x] 完成了vtrace对tc文件的可视化重放。vtrace测试正常
- [x] vtrace中在encode和transmit之间加了一条分界线，以便查看。vtrace测试正常
- [x] 云边协同实验去掉了自己的PC。vtrace对云边协同的可视化正常
  * 去掉PC的原因主要是PC的时间和其他Linux设备不同步，导致vtrace无法对tc文件进行正常的可视化
  * 关于NTP时间同步：即使配置了相同的NTP服务器而且1分钟同步一次，我自己的PC还是和树莓派、阿里云存在600ms以上的时间差。而这些设备即使不设置NTP同步，时间差也在20ms左右。所以猜测是我的电脑 或 Windows系统 或 Python的logging 的问题

下面列一下时间同步的相关资料，以备后用。

**时间差查看**

Linux设备之间的时间差可以通过命令clock-diff查看。但是Linux和Windows的时间差没找到相关工具。

安装：`sudo apt install iputils-clockdiff`

检查自己和目标主机的时间差方法如下。delta=目标主机时间-当前主机时间。

```
pi@pi4G:~ $ clockdiff 192.168.0.121
..................................................
host=192.168.0.121 rtt=1(0)ms/1ms delta=24ms/24ms Sat Dec 25 16:56:20 2021
```

参考链接：[clockdiff 命令详解](https://commandnotfound.cn/linux/1/173/clockdiff-命令)

**Linux设置时钟同步**

修改 /etc/systemd/timesyncd.conf 如下

```ini
[Time]
NTP=ntp7.aliyun.com
NTP=ntp6.aliyun.com
```

重启服务

```bash
sudo systemctl daemon-reload
sudo systemctl restart systemd-timesyncd.service
```

上述做法在树莓派上设置成功了，但是阿里云VPS上最后一行命令报错`Failed to restart systemd-timesyncd.service: Unit systemd-timesyncd.service is masked.` 所以并没有配置阿里云的时间同步。

参考链接：[配置Linux的时钟同步 - 知乎](https://zhuanlan.zhihu.com/p/308657190)

**Windows的时钟同步**

参考链接：[配置Windows实例NTP服务](https://help.aliyun.com/document_detail/51890.html)

----

- [x] ItgJob序列化时不使用稀疏编码。云边协同测试正常。添加了实验效果图

itg模式，LBC的效果图：

![image-20211225213016690](md-img/image-20211225213016690.png)

dif模式，NSC的效果图如下。对比上图可以看出，dif+NSC还是有一定效果的。

![image-20211225213226134](md-img/image-20211225213226134.png)

## 2021.12.24

- [x] 新增了tool/vtrace.py，读取tc文件并整理成特定结构体。规范了tc文件格式，相应修改了log代码。vtrace测试正常。

trace文件（.tc）格式：

每一行用空格切分必定得到4个块，分别是：时间戳，开始结束标识，动作，IFR标识。具体来说是：

```
%Y-%m-%d,%H:%M:%S {start|finish} {decode|encode|transmit|process} IFR<数字>[-finished]
```

1. 动作的起止点
   1. 动作为decode/encode/process：finish就在当前设备文件中start后面的某一行
   2. 动作为transmit：finish在下一个设备文件中的某一行
      1. 若当前设备为master：下一个设备为worker0
      2. 若当前设备为worker i且IFR未完成：下一个设备为worker i+1
      3. 若当前设备为worker i且IFR已完成：下一个设备为master
2. Master只有一个线程：`m->`负责encode+transmit。此外process的start和finish用于标识一个IFR的起始结束时间
3. Worker包含三个线程：`->w`负责decode，`w`负责execute，`w->`负责encode+transmit

## 2021.12.23

- [x] 修复了Scheduler.predict_dag参数命名的小问题
- [x] :four_leaf_clover: [实验] 在f_lnz_pred.py中添加了预测整体非零占比的OLRPredictor和OBAPredictor，和LgiPredictor配套使用
  * 添加这两个配套Predictor之后，Logistic回归的效果明显变好了，AlexNet, VGG16, ResNet50上两者基本持平
  * GoogLeNet上仍然MLP更好一些，这是可能是因为Logistic训练的不好，因为训练时报错`RuntimeWarning: overflow encountered in exp`

- [x] :four_leaf_clover: [实验] 修复了LgiPredictor在GoogLeNet上大数溢出和无法估计协方差矩阵的问题
  * 修复以后，LgiPredictor在ResNet50上的性能似乎受到了一点点影响。总体来看，AlexNet, VGG16上两者基本持平，ResNet50上MLP略好一点点，GoogLeNet上MLP更好。所以暂时仍然用MLP

- [x] 添加了trace的log，把关键事件写入tc文件以便debug。PC本地协同测试正常
  * 未来可以写一个工具读取tc文件，对整个过程进行可视化，以便直观看到瓶颈

## 2021.12.22

- [x] :four_leaf_clover: [实验] 修改了ionz_show.py，使之可以展示Logistic函数回归的结果
  * 发现Logistic函数对数据整体非零占比拟合效果相当好，可能考虑不用感知机了

- [x] :four_leaf_clover: [实验] 修改了gn197.py中读取的LFCNZ数据名称，使之与当前命名兼容
- [x] :four_leaf_clover: [实验] ionz_show.py改名为ionz_fit.py
  * show只是展示数据中呈现的规律，fit表示对数据进行拟合

- [x] :four_leaf_clover: [实验] 添加了f_lnz_pred.py，以便对比MLP预测器和Logistic预测器的效果
  * 虽然MLP看上去更好，但可能是因为Logistic预测器的cnz信息丢失导致其他预测器不准
  * 尽管MLP看上去更好，但是MLP在GoogLeNet下仍然存在预测一条直线的问题，还有改进空间

## 2021.12.21

- [x] :four_leaf_clover: [实验] 添加了lcnz_show.py，以便展示每层输出数据差值不同通道的非零占比
- [x] :four_leaf_clover: [实验] 修改了ionz_show.py，使之可以在原图上显示预测结果 
- [x] :four_leaf_clover: [实验] 修改了ionz_show.py，使之可以展示Predictor和3次函数的拟合结果
- [x] :four_leaf_clover: [实验] 修改了ionz_show.py，使之可以展示MLP直接拟合整体非零占比的结果

## 2021.12.20

- [x] Worker的LayerCost缓存文件名包含了帧大小，log做了小调整。PC本地协同测试正常
- [x] :four_leaf_clover: [实验] 添加了costs_show.py，以便展示worker的计算耗时
- [x] :four_leaf_clover: [实验] 添加了lnz_show.py，以便展示各层输出差值的稀疏情况
- [x] :four_leaf_clover: [实验] 添加了ionz_show.py，以便展示每层的输入差值非零占比和输出差值非零占比的关系
- [x] :four_leaf_clover: [实验] 添加了img_dif_show.py，以便对于两张先后的图像，展示它们在CNN各层输出数据差值的非零占比变化曲线
- [x] :four_leaf_clover: [实验] 修改了lfcnz_gen.py，直接调用Trainer相关函数生成LFCNZ数据
- [x] :four_leaf_clover: [实验] 补交img_dif_show所需的两张图像，ionz_show功能小修改

## 2021.12.15

- [x] Scheduler考虑了首个IFR和后续IFR数据传输量不同的问题，Master小修改。pipeline测试正常，PC本地协同测试正常，但是调度策略正确性有待验证
  - [x] master.py：当误差过大时输出warning
- [x] 把Scheduler改成了接口，原先的调度逻辑变成NBCScheduler。nbc_show测试正常，PC本地协同测试正常
- [x] 添加了baseline的调度器LBCScheduler。云边协同测试正常
  * 云边协同运行10帧，dif模式下性能对比：LBC每帧平均1.6s，NBC每帧平均1.55s
  * NBC比LBC效果不明显，原因可能是带宽数据不准确，而且没有考虑压缩和解压缩耗时

- [x] 把配置文件传参改成了每个类的config只有它自己的配置，而非全局配置；其他代码优化。PC本地协同测试正常
  - [x] dnn_config.py，dnn_models目录：DNNConfig添加了dnn名称
  - [x] master.py：把Scheduler初始化放到了单独的函数中
  - [x] lbc_scheduler.py：增加了对ItgJob的支持
  - [x] nsc_scheduler.py，nsc_show.py：NBC改名为NSC
  - [x] util.py：删掉了dnn_abbr函数

## 2021.12.14

- [x] bug修复：Scheduler.gen_wk_jobs返回self.__wk_jobs时要更新输入数据。PC本地协同测试误差正常

## 2021.12.12

- [x] 实现了Worker到Master的逆向通信，不再需要内网穿透了；scheduler中修复了Scheduler.optimize_chain的bug。PC本地协同测试正常，云边协同还没测试，README还没改
  - 这样做的主要原因：有可能IFR在到达最后一个Worker之前完成，所以每一个Worker都需要能够单独给Master发送report_finish，所以需要实现无需内网穿透的Master到Worker的通信渠道
  - 这样做的缺点：最后一个Worker到Master的时延无法获取了，需要通过改用时间戳来计时

- [x] Worker1已完成时直接发给Master而不发给Worker2。PC本地协同测试正常
- [x] 修改Master，使之可以接受IFR1在IFR0之前到达。云边协同测试正常
  * 这样做的原因：因为调度策略可能会发生变化，所以先发出的IFR可能后到达

- [x] 修改了README的网络配置部分
- [x] Scheduler改成了只对IFR0和IFR1进行调度，后面的直接使用IFR1的方案。云边协同测试正常
  * 这样做的原因：每次IFR调度变化都会引起缓存失效，从而导致需要传输全部数据，所以调度策略应该尽可能少地变化。IFR0的数据量和后面的显著不同，所以要单独考虑

## 2021.12.11

- [x] pipeline更新成真实数据，测试正常
- [x] 通过配置itg或dif可以选择运行模式。单元测试正常，PC本地协同测试正常
  * 启动时根据itg或dif增加配置项executor和job
  * tensor和msg互转移到了util
  * IntegralJob改名ItgJob，和DifJob都继承自Job，IFR中用Job而非DifJob

## 2021.12.10

- [x] 添加AsyncClient把new_ifr和report_finish这种无需结果的RPC改成了异步发送请求。PC本地协同测试正常
- [x] 修复了Scheduler.optimize_chain中没有考虑worker无layer的bug，config的带宽使用了真实数据，README完善。云边协同测试正常
  * 两种方法网速测试结果如下：（使用wget从python的httpserver下载；代码中的timed_rpc输出）

| 设备                    | wget测得网速          | grpc测得网速     |
| ----------------------- | --------------------- | ---------------- |
| pi2G1 (m) - pi4G (w0)   | 6.29 MB/s             | 5.34 MB/s        |
| pi4G (w0) - PC (w1)     | 4.45 MB/s             | 7.61 MB/s        |
| PC (w1) - aliyun(w2)    | 695 KB/s = 0.679 MB/s | 1.66 MB/s        |
| aliyun (w2) - pi2G1 (m) | 692 KB/s = 0.676 MB/s | 数据量太小, 未知 |

## 2021.12.9

- [x] 改变了config中的网络表示方式，所有Worker直接使用list表示，添加了对序列化和传输的计时，把通用的StubFactory改成了分Master和Worker的两种实现，配置字典统一命名为config。PC本地协同测试正常，但云边协同还没测，README还没更新
  - [x] config.yml，main.py，master.py，master_servicer.py，stub_factory.py，trainer_servicer.py，util.py，worker.py，worker_servicer.py
- [x] 修复了check=false时报错的问题，Stub加上了debug输出。PC本地协同测试正常，但云边协同还没测，README还没更新
  * 因为写涉及的文件意义不大，所以除了需要特殊说明的，之后开发笔记中每个commit不再写涉及的文件

- [x] 云边协同测试正常，更新了README。测试方案如下：

| 设备   | 服务             | 地址                                     |
| ------ | ---------------- | ---------------------------------------- |
| pi2G1  | master           | 121.199.15.155:11110 （内网穿透）        |
| pi4G   | worker0          | 192.168.0.114:33330                      |
| PC     | worker1, trainer | 114.212.84.93:33331, 114.212.84.93:22220 |
| aliyun | worker2          | 121.199.15.155:33332                     |

## 2021.12.8

- [x] README中添加了网络配置说明，config使用了基于frp内网穿透的配置。按照README的配置，通过frp内网穿透，实现了 端(Master-pi2G1)，边(Worker0-pi4G, Worker1-PC)，云(Worker2-阿里云) 的协同，测试正常
  - [x] requirements.txt：协同时会有sklearn的`UserWarning: Trying to unpickle estimator MLPRegressor from version 0.23.2 when using version 1.0.1`。 因为Trainer版本是0.23.2，但是Master版本是1.0.1，所以要把sklearn的版本统一固定在1.0.1

## 2021.12.7

- [x] 添加了requirements.txt和README.md

## 2021.12.6

- [x] Scheduler中把计算各Worker传输和计算耗时的代码独立成plan2costs_chain，pipeline的可视化函数visualize_frames移到Scheduler。pipeline测试正常(可以看到迭代优化的可视化过程)，但协同测试会崩溃(因为matplotlib只能在主线程显示)
  - [x] scheduler.py，pipeline.py
- [x] Scheduler.visualize_frames加了可视化选项，matplotlib在需要可视化时才导入，真实场景无需导入。pipeline和协同测试都正常
  - [x] scheduler.py，pipeline.py
- [x] 修复了Scheduler.simulate_pipeline的bug，即Worker2已经完成了所有层时，Worker3就不需要传输和计算了。pipeline和协同测试正常
  - [x] scheduler.py
- [x] 重构了Master，Worker，Trainer的rpc部分，IFR移动到了core，使得所有请求都通过stub_factory中的`*Stub`来序列化，通过`*Servicer`来反序列化。协同测试正常
  - [x] ifr.py：因为IFR同时被Master和Worker引用，所以把IFR从worker移动到core
  - [x] master.py，master_servicer.py，msg.proto：check_result改成report_finish，完成时总是上报，需要检查结果时上传Tensor
  - [x] scheduler.py：IFR移动，相应修改
  - [x] stub_factory.py：把Master，Worker，Trainer的发出请求过程全部集中在这里
  - [x] worker.py，worker_servicer.py：IFR移出去了，序列化相关的全部移出去了
- [x] Master新增了对IFR的计时。协同测试正常
  - [x] master.py
- [x] Master新增了配置项`pd_num`和`itv_num`，把保存PendingIpt的deque改成了线程安全的Queue。协同测试正常
  - [x] master.py，config.yml
- [x] 把print替换成logger。pipeline和协同测试正常
  - [x] config.yml：配置文件改成调试状态
  - [x] logging.yml：添加了相应配置
  - [x] master.py，master_servicer.py，scheduler.py，trainer.py，trainer_servicer.py，worker.py，worker_servicer.py：print改logger，需要logger的静态函数添加默认处理，Trainer代码风格整理
  - [x] util.py：cached_func使用logger输出，添加了缓存路径的选项
- [x] Master添加了平均IFR时延的统计。协同测试正常
  - [x] master.py
- [x] Scheduler修复bug：只对有改进的方案进行可视化。pipeline测试正常
  - [x] scheduler.py，pipeline.py

## 2021.12.5

- [x] Scheduler新增了从负载均衡方案开始局部搜索的函数optimize_chain。协同测试可以优化得到结果(正确性不确定)，但因为IntegralExecutor不支持空任务，最后一个Worker会崩溃
  - [x] scheduler.py
- [x] IntegralExecutor增加了对空任务的支持，新增了相关的单元测试。单元测试正常，协同测试正常
  - [x] integral_executor.py，test_integral_executor.py，test_dif_executor.py
- [x] Scheduler的配置中添加了带宽。协同测试正常
  - [x] config.yml，master.py，scheduler.py
- [x] Scheduler单元测试小bug修复
  - [x] test_scheduler.py
- [x] Scheduler的耗时估计单独抽象出一个函数simulate_pipeline，修复了之前的bug。使用pipeline.py对其进行可视化，但还没对ax_pc进行可视化。pipeline可视化正常，协同测试正常
  - [x] scheduler.py，pipeline.py

## 2021.12.4

- [x] 确保Worker Id从0开始连续增加，计算能力的baseline用worker0，协同运行正常
  - [x] main.py：加载config时检查
  - [x] master.py：传参数改用list
  - [x] scheduler.py：dict改成list
  - [x] config.yml：注释修改

## 2021.12.3

- [x] 新增了Scheduler.estimate_latency函数用于估计耗时，协同测试正常，但还没测试这个函数的正确性
  - [x] scheduler.py

## 2021.12.2

- [x] 添加了视频数据：campus，parking

> 视频数据来源：[The VIRAT Video Dataset](https://viratdata.org/)的release2.0
>
> | 本仓库中的视频名 | VIRAT中的视频名                 |
> | ---------------- | ------------------------------- |
> | campus           | VIRAT_S_010204_01_000072_000225 |
> | parking          | VIRAT_S_050201_05_000890_000944 |
> | road             | VIRAT_S_050000_13_001722_001766 |

## 2021.12.1

- [x] Scheduler添加了split_chain函数，用来对链状CNN按照Worker的计算能力进行均匀切割，使得耗时尽可能接近；Master执行的IFR数加入了config；DAG网页生成在当前执行路径下；其他小修改。单元测试和本地测试正常
  - [x] scheduler.py，test_scheduler.py：添加了split_chain函数，修改了初始化，添加了单元测试
  - [x] config.yml，master.py：IFR加入配置文件
  - [x] raw_dnn.py：DAG网页生成路径改到了当前路径
- [x] 对Scheduler的split_chain进行了修改，单元测试和协同测试正常
  - [x] scheduler.py，test_scheduler.py
- [x] :four_leaf_clover: [实验] lfcnz_show的sz模式添加了显示数据量预测的error

## 2021.11.29

- [x] SizedNode中添加了nz_thres，Scheduler添加了lcnz2lsz函数，实验进行了相应修改。lfcnz_show和协同运行均正常
  - [x] scheduler.py：SizedNode中添加了nz_thres，Scheduler添加了lcnz2lsz函数
  - [x] master.py：把wk_costs传入Scheduler构造函数
  - [x] lfcnz_show.py：相应修改
  - [x] config.yml：方便起见，默认配置使用480x720分辨率

## 2021.11.24

- [x] :four_leaf_clover: [实验] lfcnz_show的sz模式展示各层原始和压缩后的数据量
  - [x] experiments/lfcnz_show.py

## 2021.11.23

- [x] :four_leaf_clover: [实验] lfcnz_show.py展示三个图：误差，真实值，预测值
  - [x] experiments/lfcnz_show.py
- [x] 修复了Scheduler在对DNN进行predict时的bug：没有收集到所有前驱的数据就进行了预测。lfcnz_show在GoogLeNet上运行正常
  - [x] scheduler.py：bug修复
  - [x] lfcnz_show.py：GoogLeNet的数据集暴露了这个问题

## 2021.11.22

- [x] 使用Trainer和WorkerProfiler对离线数据收集过程进行抽象，torch==1.4.0+cpu且torchvision==0.5+cpu下master输出误差在1e-7左右

  * 注：之前误差比这个大，是因为版本较高（torch==1.9），而且这个版本下测试时执行Module会直接崩溃，所以库版本应该以 torch==1.4.0+cpu，torchvision==0.5+cpu 为准

  - [x] .gitignore：屏蔽缓存文件格式
  - [x] config.yml：添加了Trainer和WorkerProfiler相关配置
  - [x] dif_executor.py：DifExecutor使用了泛型type hint
  - [x] executor.py：Node定义移动到这里，Executor使用了泛型type hint
  - [x] integral_executor.py：IntegralExecutor使用了泛型type hint
  - [x] main.py：添加了Trainer的启动命令
  - [x] master.py：启动时添加了获取输出数据大小、获取Worker计算能力、获取Predictor的过程
  - [x] master_servicer.py：Master传参改成了直接传入全局config
  - [x] msg.proto：添加了profile相关的rpc
  - [x] node.py：因为Node功能大大简化，所以删掉了
  - [x] raw_dnn.py：把RawLayer转Node的代码移动到了Node类中
  - [x] scheduler.py：添加了对输出数据大小的profile代码，因为这个只有Scheduler会用到
  - [x] stub_factory.py：添加了Trainer的client
  - [x] trainer.py：运行在高性能服务器上，收集稀疏数据，训练预测模型
  - [x] trainer_servicer.py：为Trainer提供rpc服务
  - [x] worker.py：Worker初始化直接读取全局配置，添加了获取每层耗时的rpc接口
  - [x] worker_profiler.py：对Worker各层执行耗时进行统计
  - [x] worker_servicer.py：添加了获取每层耗时的rpc接口
  
- [x] :four_leaf_clover: experiments添加了gn197.py，用于查看GoogLeNet第197层预测不准的原因​ 

  - [x] experiments/gn197.py

- [x] [环境说明] torch1.4没有count_nonzero这个API，所以应该使用 torch==1.7.0+cpu且torchvision==0.8.1+cpu，该配置下master输出误差在1e-6到3e-6左右
* 注：这个版本是有count_nonzero的最低版本

- [x] 把代码分类整理到了多个目录中，运行正常
- [x] 把rpc相关代码整理到了目录rpc下面，测试运行正常
- [x] 把profile缓存抽象成通用的util，用于对Worker和Trainer的profile结果进行缓存
  - [x] trainer.py：cached_func移到core.util中
  - [x] util.py：通用的结果缓存函数
  - [x] worker.py：把WorkerProfiler的代码放到了这里
  - [x] worker_profiler.py：删除了
  - [x] worker_servicer.py：添加日志输出
- [x] 使用util.cached_func对Master的profile结果进行缓存，测试正常
  - [x] master.py：添加了缓存，去掉了存在误差的TODO
  - [x] scheduler.py：SizedNode.raw2dag函数重命名，以免与Node同名函数冲突
- [x] 添加了使用Predictor进行预测的代码，测试正常
  - [x] master.py：修复了没有反序列化的bug
  - [x] master_servicer.py：添加了调试输出
  - [x] scheduler.py：添加了预测代码
  - [x] worker_servicer.py：调试输出修改
- [x] :four_leaf_clover: [实验] lfcnz_show.py添加了对各层输出数据精度整体的可视化
  - [x] experiments/lfcnz_show.py
  - [x] scheduler.py：type hint的相关修改

## 2021.11.17

- [x] 使用Predictor对稀疏率预测进行抽象，去掉了LRD的部分，使用experiments中的lfcnz_show测试正常
  - [x] predictor.py：常用稀疏预测模型
  - [x] dnn_models的googlenet.py, resnet.py：做了相应修改
  - [x] experiments/lfcnz_show.py：改用了Predictor进行预测
  - [x] dnn_config.py：去掉了LRD部分，添加了Predictor
  - [x] raw_dnn.py：去掉了LRD部分
  - [x] lrd.py：删掉了
  - [x] node.py：去掉了LRD部分

## 2021.11.15

- [x] 把IntegralExecutor传入的参数从dnn_loader改成了RawDNN，Executor不再负责正确性检查，单元测试正常

## 2021.11.13

- [x] :four_leaf_clover: 添加了experiments目录，添加了LFPNZ格式数据的生成和可视化脚本
  - [x] experiments下的lfpnz_gen.py，lfpnz_show.py：生成和可视化脚本

- [x] :four_leaf_clover: ​LFPNZ数据可视化脚本添加了对DAG结构DNN的支持，GoogLeNet和ResNet50可视化正常
  - [x] experiments/lfpnz_show.py
- [x] :four_leaf_clover: experiments/lfpnz_show.py中添加了每个通道一个感知机的方案MLPs，发现：当输出通道少的时候，这个方案表现不如所有通道共用一个感知机的方法MLP，但是当输出通道多的时候（ResNet50后面的一些层），MLP效果极差，但是MLPs效果极好
- [x] :four_leaf_clover: 把关于“平面”称呼改成“通道”，本开发笔记所有experiments的提交说明都以:four_leaf_clover:开头​
- [x] msg.proto注释中的“平面”改成“通道”

## 2021.11.11

- [x] 修复了WorkerServicer的一个小bug，Master和3个Worker协同测试误差和之前一样
  - [x] scheduler.py：AlexNet在3个Worker上的执行计划
  - [x] worker_servicer.py：bug修复
  - [x] config.yml：添加了一个Worker
- [x] 把Servicer的RPC client封装成了StubFactory，Master和3个Worker协同测试误差和之前一样
  - [x] stub_factory.py：封装Client的获取过程
  - [x] master.py，master_servicer.py，worker.py，worker_servicer.py：调RPC方式相应修改

## 2021.11.10

- [x] 添加了Worker，带Scheduler的Master以及相应gRPC的Servicer，Master和1个Worker测试误差在3e-6到5e-6左右，但这个误差是Master在验证计算时带来的，不是Worker带来的
  - [x] config.yml：配置文件
  - [x] logging.yml：日志配置文件
  - [x] msg.proto：添加了Master服务和相关结构体，IFR的JobMsg改成WkJobMsg
  - [x] main.py：启动入口
  - [x] scheduler.py：只为AlexNet的一个worker生成IFR
  - [x] master.py：产生并发送IFR任务
  - [x] master_servicer.py：使用grpc提供服务
  - [x] worker.py：定义IFR，获取并执行Job
  - [x] worker_servicer.py：使用grpc提供服务

- [x] 把DNN配置参数封装成了DNNConfig，raw_layers封装成了RawDNN，单元测试全部通过，Master和1个Worker测试误差仍然在3e-6到5e-6左右
  - [x] dnn_config.py：DNNConfig所需的各种接口类
  - [x] raw_dnn.py：根据DNNConfig可以获取的原始类，可以执行，对外隐藏了不必要的API
  - [x] dnn_models/：chain.py，googlenet.py，resnet.py做了相应修改
  - [x] node.py：相应修改
  - [x] integral_executor.py：相应修改
  - [x] lrd.py：相应修改
  - [x] master.py：相应修改

## 2021.11.7

- [x] 修正了是否用CSR的判断依据，单元测试正常
  - [x] dif_executor.py：之前按照1/3判断是错的，更新了公式；应该用CSC时输出warning
  - [x] unit_tests/test_dif_executor.py：测试数据大小使用360p，非零占比为49.5%

## 2021.11.5

- [x] 添加了对DifJob的序列化，单元测试正常
  - [x] msg.proto：序列化数据格式
  - [x] msg_compile.cmd：编译proto文件
  - [x] dif_executor.py：Tensor、DifJob和pb数据转换
  - [x] unit_tests/test_dif_executor.py：添加了序列化数据转换的单元测试
  - [x] .gitignore：屏蔽了pb生成的代码

## 2021.11.3

- [x] 使用Git LFS管理视频数据
  - [x] .gitattributes：Git LFS配置文件
  - [x] media：改用Git LFS管理

## 2021.11.2

- [x] 修复了make_dag打印日志的bug和IntegralExecutor的bug，把视频放到了media目录下。DNN生成信息不再重复打印；dif_executor的fixed_jobs和var_jobs测试平均误差都在1e-6以内。
  - [x] dnn_dag.py：修复了默认logger重复打印DAG生成信息的问题
  - [x] integral_executor.py：IntegralExecutor执行完一个job要重置所有node，不能只重置job指定的node
  - [x] dif_executor.py：新增了单元测试var_jobs
  - [x] media：添加了林荫道路视频road.mp4

- [x] 把IntegralExecutor和DifExecutor的测试代码放到了单独的unit_tests目录下。测试全部通过
  - [x] integral_executor.py：去掉了在main中的测试代码
  - [x] dif_executor.py：去掉了在main中的测试代码
  - [x] unit_tests：单元测试的目录
    - [x] common.py：单元测试共用的函数，如从视频中读取一帧
    - [x] test_integral_executor.py：对IntegralExecutor的单元测试，包括各类CNN
    - [x] test_dif_executor.py：对DifExecutor的单元测试，使用差值数据，运行中worker的job会变化

## 2021.11.1

- [x] Executor作为接口，使用IntegralExecutor和DifExecutor分别执行完整的Job（IntegralJob）和仅有差值的Job（DifJob）。integral_executor单元测试误差为0，dif_executor单元测试误差在1e-6以内
  - [x] executor.py：接口定义
  - [x] dnn_dag.py：make_dag的logger参数添加了默认值
  - [x] integral_executor.py：把单元测试放进函数，RawLayer执行放入IntegralExecutor
  - [x] dif_executor.py：使用InCache和OutCache保存上一帧数据，测试了DifJob固定时执行的误差；DifJob和rpc之间的转换还没测试，所以先注释掉

## 2021.10.31

- [x] Executor添加了内存回收，使得内存占用最小，executor运行ResNet50正常
  - [x] executor.py：单元测试代码检查output非None的节点，结果正常
- [x] 去掉了单元测试代码，executor运行ResNet50正常
  - [x] executor.py：优化代码风格

## 2021.10.30

- [x] 使用了Executor，IFR，Job来对Worker执行的任务进行抽象，executor运行AlexNet正常
  - [x] chain.py：添加了prepare_vgg16
  - [x] googlenet.py：InceptionCat把列表传参改成了用`*inputs`传参数
  - [x] resnet.py：BottleneckAdd把列表传参改成了用`*inputs`传参数
  - [x] dnn_dag.py：`__make_dag`生成RawLayer后，把所有的ReLU.inplace置为False
  - [x] node.py：calc直接使用RawLayer中的calc；去除对padding的修改；去除对ReLU.inplace的修改；添加了DeepSlicing中的Master.init_dag_range，并使用init_rdag进行封装，用于初始化各Node的输入输出区间
  - [x] executor.py：Worker一次执行多个层为一个Job，一个帧对应的执行计划为一个IFR，ExNode用于保存单个层的输出数据，Executor对于一个CNN的任意Job进行执行

- [x] executor运行VGG19正常
- [x] 修复了RawLayer运行的一个小问题，executor以链状Job的方式运行GoogLeNet正常
  - [x] googlenet.py：打印出所有layer
  - [x] dnn_dag.py：修复了execute_dag中有多个输入时报错的问题
  - [x] executor.py：改用了GoogLeNet，多个Job只有一条边相连
- [x] executor以子图切割的方式运行GoogLeNet正常
  - [x] executor.py：GoogLeNet的Job之间有多条边相连
  - [x] echarts_util.py：把echarts图变大了，方便查看
- [x] executor以子图切割的方式运行ResNet50正常
  - [x] executor.py：ResNet50的Job之间有多条边相连
  - [x] resnet.py：打印出所有layer，代码风格优化
  - [x] googlenet.py：代码风格优化