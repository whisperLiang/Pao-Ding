import cv2

from core.raw_dnn import RawDNN
from core.util import cached_func
from dnn_models.chain import prepare_alexnet, prepare_vgg16
from dnn_models.googlenet import prepare_googlenet
from dnn_models.resnet import prepare_resnet50
from master.master import Master
from master.scheduler import Scheduler, SizedNode
from schedulers.nsc_scheduler import NSCScheduler
from trainer.trainer import Trainer


if __name__ == '__main__':
    CNN_NAME = 'ax'
    VIDEO_NAME = 'road'
    RESOLUTION = '480x720'
    NFRAME_TOTAL = 400
    NFRAME_SHOW = 10

    cnn_loaders = {'ax': prepare_alexnet,
                   'vg16': prepare_vgg16,
                   'gn': prepare_googlenet,
                   'rs50': prepare_resnet50}
    raw_dnn = RawDNN(cnn_loaders[CNN_NAME]())
    # AlexNet在pi4G上的耗时
    ly_comp = [0.0, 0.2958364963531494, 0.008954572677612304, 0.05184073448181152, 0.3531335830688477,
               0.0068168163299560545, 0.03638005256652832, 0.15417227745056153, 0.003465700149536133,
               0.2227616786956787, 0.0024199485778808594, 0.1506051540374756, 0.0024116992950439452,
               0.010980749130249023]
    wk_cap = [1, 1, 1]
    wk_bwth = [5.34, 7.61, 1.66]  # 单位MB
    wk_bwth = [bw*1024*1024 for bw in wk_bwth]

    wk_lynum = Scheduler.split_chain(ly_comp[1:], wk_cap)
    lb_wk_layers = Scheduler.wk_lynum2layers_chain(1, wk_lynum)

    cap = cv2.VideoCapture(f'../media/{VIDEO_NAME}.mp4')
    frame_size = tuple(map(int, RESOLUTION.split('x')))
    ipt0 = Master.get_ipt_from_video(cap, frame_size)
    ipt1 = Master.get_ipt_from_video(cap, frame_size)
    s_dag = cached_func(f"{CNN_NAME}.{RESOLUTION}.sz", SizedNode.raw2dag_sized, raw_dnn, frame_size)

    lfcnz_path = '' + CNN_NAME + '.' + VIDEO_NAME + '.' + RESOLUTION + '.' + str(NFRAME_TOTAL) + '.lfcnz'
    print("collecting LFCNZ data...")
    lfcnz = cached_func(lfcnz_path, Trainer.collect_lfcnz, raw_dnn, f'../media/{VIDEO_NAME}.mp4',
                        NFRAME_TOTAL, frame_size)
    pred_path = CNN_NAME + '.' + VIDEO_NAME + '.' + RESOLUTION + '.' + str(NFRAME_TOTAL) + '.pred'
    print("training predictors...")
    predictors = cached_func(pred_path, Trainer.train_predictors, raw_dnn, lfcnz)

    # TODO: 使用动画展示迭代过程
    # 第0帧调度
    opt_lbsz = NSCScheduler.dif2lbsz(ipt0, s_dag, predictors)
    wk_elys = NSCScheduler.optimize_chain(lb_wk_layers, [[] for _ in range(len(wk_cap))], wk_cap, wk_bwth,
                                          opt_lbsz, opt_lbsz, ly_comp, 1, True)
    # 第1-5帧调度
    pre_wk_ilys = [([lys[0]] if len(lys) > 0 else []) for lys in wk_elys]
    opt_lbsz = Scheduler.dif2lbsz(ipt1, s_dag, predictors)
    dif_lbsz = Scheduler.dif2lbsz(ipt1-ipt0, s_dag, predictors)
    NSCScheduler.optimize_chain(lb_wk_layers, pre_wk_ilys, wk_cap, wk_bwth, opt_lbsz, dif_lbsz, ly_comp, 10, True)