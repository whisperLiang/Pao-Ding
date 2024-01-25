from core.raw_dnn import RawDNN
from dnn_models.faster_rcnn import prepare_fasterrcnn
from worker.worker import Worker

from matplotlib import pyplot as plt

from itertools import accumulate

raw_dnn = RawDNN(prepare_fasterrcnn())
layer_costs = Worker.profile_dnn_cost(raw_dnn, (480,720), 5)
print(layer_costs)
plt.plot(layer_costs)

for i, cost in enumerate(layer_costs):
    if cost > 0.1:
        print(i, cost)

acc = list(accumulate(layer_costs))
plt.figure()
plt.plot(acc)
plt.show()