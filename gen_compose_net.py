#!/usr/bin/env python3

import sys
import yaml

def generate_worker_config(ip):
    """Generate configuration for a single worker."""
    return f"""
  paoding-worker{ip}:
    <<: *paoding-common
    cpuset: "{ip//4 + 10}"
    command: python3 main.py worker -i {ip}
    networks:
      paoding-network:
        ipv4_address: 174.28.0.{ip+3}
    healthcheck:
      test: ["CMD-SHELL", "curl --silent --fail http://localhost:8000 || exit 1"]
      interval: 7s
      timeout: 5s
      retries: 100
"""

def generate_docker_compose(num_workers):
    """Generate the complete Docker Compose YAML content."""
    base_services = """
version: '3.8'
x-paoding-common: &paoding-common
  image: pao-ding:1.0
  volumes:
    - /home/yons/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints/
    - /home/yons/whisperliang/Pao-Ding:/Pao-Ding
  deploy:
    resources:
      limits:
        cpus: "0.25"
        memory: 2G  

networks:
  paoding-network:
    driver: bridge
    ipam:
      config:
        - subnet: 174.28.0.0/24

services:
  paoding-master-trainer:
    image: pao-ding:1.0
    cpuset: "8-9"
    volumes:
      - /home/yons/.cache/torch/hub/checkpoints:/root/.cache/torch/hub/checkpoints/
      - /home/yons/whisperliang/Pao-Ding:/Pao-Ding
    command: python3 main.py trainer
    networks:
      paoding-network:
        ipv4_address: 174.28.0.2
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
    healthcheck:
      test: ["CMD-SHELL", "curl --silent --fail http://localhost:8000 || exit 1"]
      interval: 7s
      timeout: 5s
      retries: 100
"""

    # Generate worker configurations
    worker_configs = "".join(generate_worker_config(i) for i in range(num_workers-1, -1, -1))

    # Combine base services with worker configurations
    full_compose = base_services + worker_configs
    return full_compose

def generate_net_config(new_yaml_file_path, workers_num, bw):
    # 初始化net配置
    net_config = {}
    bandwidth = [bw] * workers_num
    
    # 添加master到其他节点的网络配置（示例配置）
    net_config['m->t'] = f'174.28.0.2:11112'
    for i in range(workers_num):
        net_config[f'm->w{i}'] = f'174.28.0.{i+3}:11113'
    
    # 添加worker间的网络配置
    for i in range(workers_num - 1):
        from_worker = f'w{i}'
        to_worker = f'w{i+1}'
        net_config[f'{from_worker}->{to_worker}'] = f'174.28.0.{i+4}:11113'
    
    # 将net配置保存到新的YAML文件
    with open(new_yaml_file_path, 'w') as file:
        yaml.safe_dump({'net': net_config}, file)
        yaml.safe_dump({'bandwidth': bandwidth}, file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_workers.py <num_workers>")
        sys.exit(1)

    num_workers = int(sys.argv[1])

    temp_compose_path = "temp_docker-compose.yml"
    net_path = "net.yml"
    bw = 10
    compose_content = generate_docker_compose(num_workers)
    generate_net_config(net_path, num_workers, bw)

    with open(temp_compose_path, "w") as file:
        file.write(compose_content)

    print(f"Generated Docker Compose configuration at {temp_compose_path}")
    print(f"Generated net configuration at {net_path}")