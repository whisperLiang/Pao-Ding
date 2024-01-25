import matplotlib.pyplot as plt
import numpy as np
plt.rc('font',family='Times New Roman')
import matplotlib
matplotlib.rc('pdf', fonttype=42)


plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
lg = {'size': 17}
leg = {'size': 14}
y_la='Encoding Time (ms)'

ITER_MARKER = '--o'
_3TO2H_MARKER = '--*'
_3TO2V_MARKER = '-x'


def draw_p():
    iter_bsr = [514.2679129, 1131.2717440000001, 1694.4019637000001, 2256.0773395, 2803.4873666000003,
                3314.6553931000003, 4567.9010397, 5175.849265000001, 5730.8223739000005, 7727.202162300001]
    iter_csr = [495.40513739999994, 1018.891733, 1496.1406092, 1966.254868, 2425.1420055999997, 2842.0152969,
                3246.7878142, 3572.2245164, 3887.7718646999992, 5365.035179699999]
    _3to2h_bsr = [911.2241203, 1708.0804026, 2482.4865220999995, 3249.3109326, 4003.9445309000002, 4729.052917799999,
                  5383.951185399999, 8204.991984100001, 8841.3537653, 9984.134079799998]
    _3to2h_csr = [909.2844175999999, 1529.7841607999999, 2121.1814342000002, 2711.3768067, 3289.0185401000003,
                  3829.8416964000003, 4311.1790666, 4751.131297400001, 5177.3988157, 5609.8471954]
    _3to2v_bsr = [884.6467629000001, 1678.4729845000002, 2436.4705077, 3186.6172465, 3922.4325785, 4640.550235199999,
                  5275.212149000001, 7241.854435300001, 7223.121025400001, 8205.9242495]
    _3to2v_csr = [887.6529022000001, 1503.1914737000002, 2101.8573100000003, 2688.7728491, 3268.6863031000003,
                  3811.3360396999997, 4296.1542258, 4732.7693519, 5158.6533147, 5588.2725465]
    # plt.plot(np.arange(0, 1, .1), iter_bsr, "-o", label="iter_bsr")
    plt.plot(np.arange(0, 1, .1), iter_csr, ITER_MARKER, label="PI")#Path Independent
    # plt.plot(np.arange(0, 1, .1), _3to2h_bsr, "-o", label="3to2h_bsr")
    plt.plot(np.arange(0, 1, .1), _3to2h_csr, _3TO2H_MARKER, label="HC")#Horizontal Combination
    # plt.plot(np.arange(0, 1, .1), _3to2v_bsr, "-o", label="3to2v_bsr")
    plt.plot(np.arange(0, 1, .1), _3to2v_csr, _3TO2V_MARKER, label="VC")#Vertical Combination
    plt.gca().set_xlabel("Non-Zero Rate", fontproperties=lg)
    plt.gca().set_ylabel(y_la, fontproperties=lg)
    plt.tick_params(labelsize=15)
    plt.legend(loc='upper left', prop=leg)
    plt.tight_layout()


def draw_D():
    iter_bsr = [249.60127110000002, 531.4583700999999, 823.3317278999999, 1117.4330178000002, 1416.3849628,
                1703.5420806999998, 1953.9217743000002, 2238.3540434, 2509.2404741]
    iter_csr = [265.67339869999995, 501.0855312999999, 760.0576785, 1018.8062168999999, 1273.8402973,
                1557.3771457000003, 1784.3823822, 2024.1035962, 2270.5174414000003]
    _3to2h_bsr = [444.51024449999994, 887.2915297, 1333.4375708000002, 1777.9343652, 2229.0324473, 2676.4080621000003,
                  3121.5806608000003, 3573.5958359000006, 4047.1104904]
    _3to2h_csr = [399.7927114, 797.8300212, 1193.1413883, 1591.5324697, 2011.8088642999999, 2411.7201698,
                  2803.6725484999997, 3187.527067, 3606.9565493]
    _3to2v_bsr = [439.1837113, 875.1401392, 1311.0539175, 1748.5204969, 2137.0704751000003, 2567.5079903, 2979.7393061,
                  3410.4420599000005, 3876.7717739999994]
    _3to2v_csr = [395.4129986, 785.1701634, 1174.9534511000002, 1575.5064032, 1909.3214927000001, 2290.1350641999998,
                  2662.3443172999996, 3047.6361454999997, 3421.8959206]
    # plt.plot(range(50, 500, 50), iter_bsr, "-o", label="iter_bsr")
    plt.plot(range(50, 500, 50), iter_csr, ITER_MARKER, label="PI")
    # plt.plot(range(50, 500, 50), _3to2h_bsr, "-o", label="3to2h_bsr")
    plt.plot(range(50, 500, 50), _3to2h_csr, _3TO2H_MARKER, label="HC")
    # plt.plot(range(50, 500, 50), _3to2v_bsr, "-o", label="3to2v_bsr")
    plt.plot(range(50, 500, 50), _3to2v_csr, _3TO2V_MARKER, label="VC")
    plt.gca().set_xlabel("Number of Paths", fontproperties=lg)
    #plt.gca().set_ylabel(y_la, fontproperties=lg)
    plt.tick_params(labelsize=15)
    plt.legend(prop=leg)
    plt.tight_layout()


def draw_R():
    iter_bsr = [293.1666859, 483.1984144, 681.960512, 849.0010523999999, 1003.7189646, 1261.1107290999998, 1345.5373057000002, 1554.3930607999998, 1767.9710495]
    iter_csr = [264.3541049, 431.8855482999999, 606.0076553, 774.6287101, 944.6459616000002, 1106.6240388,
                1270.5632988, 1430.9638469000001, 1590.4918237999998]
    _3to2h_bsr = [299.9386488, 606.0391678999999, 950.9379759, 1226.8933656, 1503.9000838, 1922.9285894, 2101.7641901,
                  2437.7297147, 2830.9352506]
    _3to2h_csr = [281.3798973, 560.9396538, 850.0579431, 1133.9937530000002, 1413.7646187, 1693.4029564, 1976.1230593,
                  2259.1547318999997, 2531.6791316000003]
    _3to2v_bsr = [305.49230259999996, 606.6360291, 914.6250206000001, 1216.4615718999999, 1518.5868589000002,
                  1821.8185407, 2127.7008136, 2425.9642592000005, 2741.9200859]
    _3to2v_csr = [273.1556233, 542.4088942000001, 817.0015059, 1087.9993588, 1358.4270155, 1629.5434403,
                  1900.7759818999998, 2171.2712865000003, 2454.4288203]

    # plt.plot(range(50, 500, 50), iter_bsr, "-o", label="iter_bsr")
    plt.plot(range(50, 500, 50), iter_csr, ITER_MARKER, label="PI")  #分别编码
    # plt.plot(range(50, 500, 50), _3to2h_bsr, "-o", label="3to2h_bsr")
    plt.plot(range(50, 500, 50), _3to2h_csr, _3TO2H_MARKER, label="HC") #横向合并
    # plt.plot(range(50, 500, 50), _3to2v_bsr, "-o", label="3to2v_bsr")
    plt.plot(range(50, 500, 50), _3to2v_csr, _3TO2V_MARKER, label="VC")#纵向合并
    plt.gca().set_xlabel("Number of Rows", fontproperties=lg)
    plt.gca().set_ylabel(y_la, fontproperties=lg)
    plt.tick_params(labelsize=15)
    plt.legend(prop=leg)
    plt.tight_layout()


def draw_C():
    iter_bsr = [229.30016, 319.3609432, 427.36274219999996, 504.7221237, 597.1267755, 728.8089160999999, 783.3167803,
                882.7712281, 1016.0823551]
    iter_csr = [188.7734323, 258.38719340000006, 355.1238134, 436.0471224, 531.9208256, 623.2337181, 717.9642139,
                810.1485435, 906.6988856]
    _3to2h_bsr = [148.9057422, 311.2224818, 495.27016460000004, 679.3691086, 833.9181581999999, 1048.3367796,
                  1255.2853658, 1413.7826019000001, 1589.1283977999997]
    _3to2h_csr = [138.4058217, 288.15575060000003, 458.9439412, 615.0528959999999, 752.8938338, 948.7121222999998,
                  1139.7996254000002, 1280.8057191, 1438.9188224]
    _3to2v_bsr = [130.4158402, 276.6400618, 448.48164479999997, 607.7335662, 751.7606936, 989.4872646,
                  1142.5902767999999, 1326.4841157, 1508.6916831]
    _3to2v_csr = [124.88693859999998, 257.3739435, 416.28798900000004, 570.2317406, 718.3256139, 889.7008889000001,
                  1067.3538901, 1210.8238700999998, 1357.7247321]
    # plt.plot(range(50, 500, 50), iter_bsr, "-o", label="iter_bsr")
    plt.plot(range(50, 500, 50), iter_csr, ITER_MARKER, label="PI")
    # plt.plot(range(50, 500, 50), _3to2h_bsr, "-o", label="3to2h_bsr")
    plt.plot(range(50, 500, 50), _3to2h_csr, _3TO2H_MARKER, label="HC")
    # plt.plot(range(50, 500, 50), _3to2v_bsr, "-o", label="3to2v_bsr")
    plt.plot(range(50, 500, 50), _3to2v_csr, _3TO2V_MARKER, label="VC")
    plt.gca().set_xlabel("Number of Columns", fontproperties=lg)
    #plt.gca().set_ylabel(y_la, fontproperties=lg)
    plt.tick_params(labelsize=15)
    plt.legend(prop=leg)
    plt.tight_layout()

if __name__ == '__main__':
    fig = plt.figure(figsize=(7, 5))
    plt.subplot(221)
    draw_p()

    plt.subplot(222)
    draw_D()

    plt.subplot(223)
    draw_R()

    plt.subplot(224)
    draw_C()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)

    plt.show()


