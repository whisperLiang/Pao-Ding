"""读取一个模型流水线执行的记录文件，显示w0到w1的中间特征传输时间
一个子图展示一个模型，横轴为帧号，纵轴为传输时间（单位S）
"""
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, TextIO, Union

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
lg = {'size': 16}
plt.rc('font',family='Times New Roman')
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.rc('pdf', fonttype=42)

@dataclass
class Event:
    ifr_id: int  # IFR号
    ifr_fin: bool  # 此IFR是否已完成
    act: str  # 事件名称
    timestamp: datetime  # 时间戳
    is_start: bool  # start还是finish

def read_events(tcfile: Union[str, TextIO], ifr_num: int = -1) -> List[List[Event]]:
    """从某个设备生成的tc文件中读取事件
    :param tcfile tc文件名，或者file-like对象
    :param ifr_num 所有事件中的IFR数量(id从0计数到ifr_num-1)。若未知可不填
    :return i_evts i_evts[i]对应id=i的IFR所有事件，事件按照时间顺序排序
    """
    events = []
    ifr_cnt = 0
    if isinstance(tcfile, str):
        tcfile = open(tcfile, 'r')
    with tcfile:
        for line in tcfile:
            timestamp, is_start, act, ifr = line[:-1].split(' ')
            timestamp = datetime.fromisoformat(timestamp)
            is_start = (is_start == 'start')
            ifr_id = int(ifr[3:].replace('-finished', ''))
            ifr_fin = ('-finished' in ifr)
            ifr_cnt = max(ifr_cnt, ifr_id+1)
            events.append(Event(ifr_id, ifr_fin, act, timestamp, is_start))
    i_evts = [[] for _ in range(max(ifr_cnt, ifr_num))]
    for event in events:
        i_evts[event.ifr_id].append(event)
    return i_evts

@dataclass
class Stage:
    act: str  # 事件名称
    thread: str  # 此阶段所属的设备线程：m->, ->w0, w0, w0->, ->w1, w1, w1->, ...
    start: datetime  # 起始时间
    finish: datetime  # 结束时间

    def __repr__(self):
        return f"Stage('{self.trans_thread(self.thread)}':{self.act}," \
               f"s='{self.start.isoformat(' ', timespec='milliseconds')}'," \
               f"f='{self.finish.isoformat(' ', timespec='milliseconds')}')"

    @staticmethod
    def trans_thread(thread: str):
        return thread.replace('$', '').replace(r'\rightarrow', '->').replace('_', '').replace(' ', '')

@dataclass
class IFRRecord:
    ifr_id: int
    start: datetime
    finish: datetime
    stages: List[Stage]

    def __str__(self):
        return f"IFRRecord(ifr={self.ifr_id}, start='{self.start.isoformat(' ', timespec='milliseconds')}', " \
               f"finish='{self.finish.isoformat(' ', timespec='milliseconds')}', " \
               f"stages={self.stages})"

def events2records(mi_evts: List[List[Event]], w_i_evts: List[List[List[Event]]],
                   act2trd: Dict[Tuple[str, str], str]) -> List[IFRRecord]:
    ircds = []
    for ifr_id in range(len(mi_evts)):
        m_evts = mi_evts[ifr_id]  # Master中当前IFR的所有事件
        # 从Master的事件中找到当前IFR的开始结束时间，并删除该事件
        se, fe = -1, -1
        for e, evt in enumerate(m_evts):
            if evt.act == 'process':
                if evt.is_start:
                    assert se < 0
                    se = e
                else:
                    assert fe < 0
                    fe = e
        start, finish = m_evts[se].timestamp, m_evts[fe].timestamp
        m_evts.pop(fe)  # 这里要先删除后面的finish事件，这样se的索引不会变
        m_evts.pop(se)
        # 根据事件，生成Stage
        ircd = IFRRecord(ifr_id, start, finish, [])
        d_evts = [m_evts] + [(i_evts[ifr_id] if i_evts else []) for i_evts in w_i_evts]  # 设备->当前IFR的所有事件
        d_name = ['$m$'] + [f'$w_{w}$' for w in range(len(w_i_evts))]  # d->设备名
        tr_sevt: Optional[Event] = None  # 最近一个传输start事件
        tr_sd: int = -1  # 最近一个传输start事件对应的设备
        for d, evts in enumerate(d_evts):  # 遍历各设备
            if d == 0:  # Master至少有一个事件，且第一个事件应该是start
                assert len(evts) > 0 and evts[0].is_start
                e = 0
            else:
                if len(evts) == 0:  # Worker上一个事件也没有，直接跳过
                    continue
                # 如果Worker上有事件，第一个事件应该是传输完成事件
                assert evts[0].act == 'transmit' and not evts[0].is_start
                s_evt, f_evt = tr_sevt, evts[0]
                ircd.stages.append(Stage(s_evt.act, act2trd[d_name[tr_sd], s_evt.act],
                                         s_evt.timestamp, f_evt.timestamp))
                e = 1
            while e+1 < len(evts):
                s_evt, f_evt = evts[e], evts[e+1]
                ircd.stages.append(Stage(s_evt.act, act2trd[d_name[d], s_evt.act], s_evt.timestamp, f_evt.timestamp))
                e += 2
            assert e+1 == len(evts)  # len(evts)-1应该是传输开始事件
            assert evts[-1].act == 'transmit' and evts[-1].is_start
            tr_sevt = evts[-1]
            tr_sd = d
        # 最后一个传输事件的finish时间为此IFR的finish时间，即Master收到的时间
        ircd.stages.append(Stage(tr_sevt.act, act2trd[d_name[tr_sd], tr_sevt.act], tr_sevt.timestamp, ircd.finish))
        ircds.append(ircd)
    return ircds

def predict_show(v1: List[float], v2: List[float]):

    x=[i for i in range(len(v1))]
    plt.plot(x,v1, '--',linewidth=2,color='magenta',label='Pao-Ding LBS',zorder=100) #原始数据
    plt.plot(x,v2, '-',color='darkorange',linewidth=2,label='Pao-Ding DAs',zorder=100) #差值数据实际值
    plt.fill_between(x, v1, v2, color='b', hatch = '//',alpha=0.5, label='Transmit Time Reduction', zorder=112)
    plt.xlabel('Index of Frame',fontsize=18)
    plt.ylabel('Transmit Time (s)',fontsize=18)
    # plt.legend(bbox_to_anchor=(0.25, 0.7),fontsize=16)
    plt.legend(loc='best',fontsize=16)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(axis="y", linestyle='-.', zorder=0)

    # plt.show()
    plt.savefig(f'trans_{dnn_name}.png', bbox_inches='tight', pad_inches=0)

def read_from_folder(folder_name: str) -> Tuple[List[List[Event]], List[List[List[Event]]]]:
    """从指定的文件夹中读取mi_evts, w_i_evts"""
    files = os.listdir(folder_name)
    # 找到master.tc文件
    master_file = [file for file in files if file == 'master.tc'][0]
    # 从master.tc读取mi_evts
    with open(os.path.join(folder_name, master_file), 'r') as m_tcfile:
        mi_evts = read_events(m_tcfile)
    
    w_i_evts = []
    # 找到worker*.tc文件并逐个读取
    for file in files:
        if file.startswith('worker') and file.endswith('.tc'):
            with open(os.path.join(folder_name, file), 'r') as w_tcfile:
                w_i_evts.append(read_events(w_tcfile, len(mi_evts)))
    
    return mi_evts, w_i_evts

def show_ifr_transtime(ifr_records1: List[IFRRecord], ifr_records2: List[IFRRecord]):
    """trds为各设备的线程，按照执行顺序排列"""
    itg_trans = []
    # all_start = ifr_records1[0].start  # 最开始的时间
    for ifr_rcd in ifr_records1:
        l=len(ifr_rcd.stages)
        #for stage in ifr_rcd.stages:
        for i in range(l):
            stage=ifr_rcd.stages[i]
            print(f'records1: {stage}')
            stage_time = (stage.finish - stage.start).total_seconds()
            print(f'{stage_time}')
            # 记录w0传输时间
            if stage.thread == '$w_0\\rightarrow$' and stage.act == 'transmit':
                itg_trans.append(stage_time)

    das_trans = []
    # all_start = ifr_records2[0].start  # 最开始的时间
    for ifr_rcd in ifr_records2:
        l=len(ifr_rcd.stages)
        #for stage in ifr_rcd.stages:
        for i in range(l):
            stage=ifr_rcd.stages[i]
            print(f'records2: {stage}')
            stage_time = (stage.finish - stage.start).total_seconds()
            print(f'{stage_time}')
            # 记录w0传输时间
            if stage.thread == '$w_0\\rightarrow$' and stage.act == 'transmit':
                das_trans.append(stage_time)
    av_itg = sum(itg_trans)/len(itg_trans)
    av_das = sum(das_trans)/len(das_trans)
    down_rate = (av_itg-av_das)/av_itg
    print(f'传输时间下降{down_rate}')
    predict_show(itg_trans, das_trans)


if __name__ == '__main__':
    # 3种模式：
    #   l: local模式，从 LOCAL_DIR 指定的目录下寻找tc文件，根据目录下的文件名判断worker数
    #   r: remote模式，从 REMOTE_CFG 获取远程服务器配置和worker数，下载远程tc文件
    #   z: zip模式，从 TCZIP 指定的zip压缩包中读取tc文件，根据压缩包中的文件名判断worker数
    MODE = 'z'
    XLIM = None  # 横轴的最大时间, None为matplotlib自动决定, 非None时最小时间也会设置为0
    LOCAL_DIR = 'lbc2'  # l模式下, 本地目录路径
    REMOTE_CFG = 'device.yml'  # 远程服务器的配置文件
    dnn_name = 'regnet'

    # TCZIP = './Pipeline_Execution_result/itg_alexnet.zip'  # 从zip文件中读取tc文件
    # g_mi_evts, g_w_i_evts = read_from_zip(TCZIP)

    TCFOLDER1 = f'./Pipeline_Execution_result/itg_{dnn_name}'  # 从文件夹中读取tc文件
    g_mi_evts, g_w_i_evts = read_from_folder(TCFOLDER1)

    print(f"events read succeeded, n_worker={len(g_w_i_evts)}, n_ifr={len(g_mi_evts)}")

    TRD2ACTS = {r'$m\rightarrow$': ['encode', 'transmit']}
    #for wid in range(len(g_w_i_evts)):
    TRD2ACTS[rf'$\rightarrow w_{0}$'] = ['decode']
    TRD2ACTS[f'$w_{0}$'] = ['execute']
    TRD2ACTS[rf'$w_{0}\rightarrow$'] = ['encode', 'transmit']
    TRD2ACTS[rf'$\rightarrow w_{1}$'] = ['decode']
    TRD2ACTS[f'$w_{1}$'] = ['execute']
    TRD2ACTS[rf'$w_{1}\rightarrow$'] = ['encode', 'transmit']

    ACT2TRD = {}  # (m, decode): 'm->', (w0, decode): '->w0'
    for trd, acts in TRD2ACTS.items():
        for act in acts:
            ACT2TRD[trd.replace(r'\rightarrow', '').replace(' ', ''), act] = trd

    g_ircds1 = events2records(g_mi_evts, g_w_i_evts, ACT2TRD)
    total_transmit = 0  # 传输总耗时
    for ircd in g_ircds1:
        print(ircd)
        total_transmit += sum((stg.finish - stg.start).total_seconds() for stg in ircd.stages if stg.act == 'transmit')
    print(f"total_transmit={total_transmit}s")



    TCFOLDER2 = f'./Pipeline_Execution_result/paoding_{dnn_name}'  # 从文件夹中读取tc文件
    g_mi_evts, g_w_i_evts = read_from_folder(TCFOLDER2)

    print(f"events read succeeded, n_worker={len(g_w_i_evts)}, n_ifr={len(g_mi_evts)}")

    TRD2ACTS = {r'$m\rightarrow$': ['encode', 'transmit']}
    #for wid in range(len(g_w_i_evts)):
    TRD2ACTS[rf'$\rightarrow w_{0}$'] = ['decode']
    TRD2ACTS[f'$w_{0}$'] = ['execute']
    TRD2ACTS[rf'$w_{0}\rightarrow$'] = ['encode', 'transmit']
    TRD2ACTS[rf'$\rightarrow w_{1}$'] = ['decode']
    TRD2ACTS[f'$w_{1}$'] = ['execute']
    TRD2ACTS[rf'$w_{1}\rightarrow$'] = ['encode', 'transmit']

    ACT2TRD = {}  # (m, decode): 'm->', (w0, decode): '->w0'
    for trd, acts in TRD2ACTS.items():
        for act in acts:
            ACT2TRD[trd.replace(r'\rightarrow', '').replace(' ', ''), act] = trd

    g_ircds2 = events2records(g_mi_evts, g_w_i_evts, ACT2TRD)
    total_transmit = 0  # 传输总耗时
    for ircd in g_ircds2:
        print(ircd)
        total_transmit += sum((stg.finish - stg.start).total_seconds() for stg in ircd.stages if stg.act == 'transmit')
    print(f"total_transmit={total_transmit}s")

    show_ifr_transtime(g_ircds1, g_ircds2)


