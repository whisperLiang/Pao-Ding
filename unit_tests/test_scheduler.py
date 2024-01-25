from master.scheduler import Scheduler


def test_split_chain():
    assert Scheduler.split_chain([2, 1, 3], [1, 2]) == [1, 2]
    assert Scheduler.split_chain([2, 1, 1, 3, 4], [1, 2]) == [3, 2]
    assert Scheduler.split_chain([2, 1, 1, 3, 4], [1, 2, .1]) == [3, 2, 0]
    assert Scheduler.split_chain([2, 1, 1, 3, 4], [.1, 1, 2]) == [0, 3, 2]
    # AlexNet在PC上的耗时（去除第0层）
    ax_pc = [0.017154693603515625, 0.002193784713745117, 0.010571908950805665,
             0.01097102165222168, 0.000997018814086914, 0.006582069396972656,
             0.006582736968994141, 0.0005986213684082032, 0.008776283264160157,
             0.000399017333984375, 0.005983781814575195, 0.0007979393005371094, 0.0011966705322265625]
    assert Scheduler.split_chain(ax_pc, [1, 1, 1]) == [2, 4, 7]


def test_wk_lynum2layers_chain():
    assert Scheduler.wk_lynum2layers_chain(1, [1, 2, 3]) == [[1], [2, 3], [4, 5, 6]]
    assert Scheduler.wk_lynum2layers_chain(1, [0, 2, 3]) == [[], [1, 2], [3, 4, 5]]
    assert Scheduler.wk_lynum2layers_chain(1, [2, 0, 3]) == [[1, 2], [], [3, 4, 5]]
    assert Scheduler.wk_lynum2layers_chain(1, [2, 3, 0]) == [[1, 2], [3, 4, 5], []]
