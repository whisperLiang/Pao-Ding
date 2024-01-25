import logging.config
from logging import LogRecord
from typing import Dict, Any

import click
import humanfriendly.terminal
import yaml

from core.dif_executor import DifExecutor, DifJob
from core.itg_executor import ItgExecutor, ItgJob
from master.master_servicer import MasterServicer
from trainer.trainer_servicer import TrainerServicer
from worker.worker_servicer import WorkerServicer


class TraceFilter(logging.Filter):
    """用于生成trace文件"""
    def filter(self, record: LogRecord) -> bool:
        return getattr(record, 'trace', False)


def config_common(config_file: str, service_name: str) -> Dict[str, Any]:
    """设置启动LocalSimulator, MasterController, WorkerController前都需要配置的参数
    :param config_file 配置文件路径
    :param service_name 服务名称，用于配置log导出的trace文件
    :return 从配置文件读取出的配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.Loader)  # 加载yaml中规定的类
    exec_suite = {
        'itg': {'executor': ItgExecutor, 'job': ItgJob},
        'dif': {'executor': DifExecutor, 'job': DifJob}
    }
    config.update(exec_suite[config['mode']])
    # 载入logger配置，因为可以加注释所以用了yaml格式，因为注释有中文所以读取编码限制为utf-8
    humanfriendly.terminal.enable_ansi_support()  # 增加对Windows彩色输出的支持
    with open(config['log_cfg'], 'r', encoding='utf-8') as f:
        log_config = yaml.load(f, yaml.Loader)
        log_config['handlers']['trace']['filename'] = service_name + '.tc'
        logging.config.dictConfig(log_config)  # 载入logger配置
    logger = logging.getLogger(__name__)
    logger.info(f"config: {config}")
    return config


@click.group()
def start_service():
    """Choose master/worker/simulator to start"""


@start_service.command(name='master')
@click.option('-c', '--config', 'config_file', type=click.Path(),
              required=False, default='config.yml', show_default=True,
              help='Configuration file path of this project')
def start_master(config_file: str):
    """Start master with desired configuration"""
    config = config_common(config_file, 'master')
    MasterServicer(config)


@start_service.command(name='worker')
@click.option('-c', '--config', 'config_file', type=click.Path(),
              required=False, default='config.yml', show_default=True,
              help='Configuration file path of this project')
@click.option('-i', '--id', 'id_', type=int, required=True,
              help='The id of this worker, which should already be in the configuration file')
def start_worker(config_file: str, id_: int):
    """Start a worker with specified id and desired configuration"""
    config = config_common(config_file, f'worker{id_}')
    WorkerServicer(id_, config)


@start_service.command(name='trainer')
@click.option('-c', '--config', 'config_file', type=click.Path(),
              required=False, default='config.yml', show_default=True,
              help='Configuration file path of this project')
def start_trainer(config_file):
    """Start a trainer with desired configuration"""
    config = config_common(config_file, 'trainer')
    TrainerServicer(config)


if __name__ == '__main__':
    start_service()
