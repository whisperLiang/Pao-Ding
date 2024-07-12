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

## 实际网络环境

### 运行
具体运行方式可以参考ResMap的README.md：[ResMap](https://github.com/nju-cn/ResMap)
主要是需要关注拓扑结构中的节点的启动顺序以及config.yml文件处的修改

## Docker仿真环境

### 工作流
为了支持任意数量的workers 和不同的网络带宽，我们在cpu核数为32，内存为64GB的服务器上基于Docker 容器实现了一套协同推理的仿真环境。
我们可以用docker compose 实现对各个workers 容器节点使用的资源（CPU 和 memory）进行配置，并且可以借助Pumba 进行容器间网络环境的仿真。
### 运行
#### 自定义配置
你可以参考docker-compose.yml文件，修改其中的配置，然后启动资源自定义的docker容器节点
```bash
docker-compose up -d
```
然后你可以通过各节点的健康检查来确认是否准备就绪，如果就绪，就可以通过Pumba进行网络带宽，延迟，丢包等设置。
```bash
pumba netem --duration 2m --tc-image gaiadocker/iproute2 rate --rate 32mbit re2:^pao-ding
```
我们通过上述脚本将以pao-ding开头的容器节点的带宽设置为32Mbps，持续时间为2min。
然后我们进入到master-trainer节点，启动master进行最终的调度进行流水线推理即可。
```bash
docker exec -it pao-ding_paoding-master-trainer_1 /bin/bash
python3 main.py master
```
如果推理完成，你可以通过docker-compose down 停止容器。
#### 自动生成配置
你可以通过gen_compose_net.py 脚本自动生成temp_docker-compose.yml文件和net.yml。
```bash
python3 gen_compose_net.py <workers_num>  
```
生成之后你需要将net.yml文件中字段复制到config.yml文件中对应的字段中，同时记得修改workers_num字段。
最后你可以通过docker-compose -f temp_docker-compose.yml up -d启动自定义数量的容器节点。
其余操作你可以参考自定义配置的操作。
需要注意的是，如果容器节点启动失败，你需要自己手动修改temp_docker-compose.yml文件中的资源配置。
