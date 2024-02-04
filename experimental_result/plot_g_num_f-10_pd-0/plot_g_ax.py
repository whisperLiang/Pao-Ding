from matplotlib import pyplot as plt
import  random
import numpy as np
plt.rc('font',family='Times New Roman')
import matplotlib
matplotlib.rc('pdf', fonttype=42)
def auto_text(rects):
    for rect in rects:
        plt.text(rect.get_x()+0.0251, rect.get_height()+0.0516, rect.get_height(), ha='left',fontsize=14, va='bottom')

fig,ax=plt.subplots(figsize=(5,4))
width = 0.35
x=[1,2,3,4,5,6,7,8,9,10]
lbs_road = [random.uniform(1.59,1.61) for i in range(10)]
das_road=[1.59,1.34,1.22,1.14,1.15,1.17,1.17,1.14,1.09,1.17]
lbs_parking = [random.uniform(1.59,1.61) for i in range(10)]
das_parking=[1.59,1.34,1.22,1.14,1.15,1.17,1.17,1.14,1.09,1.17]
error1=[0.049,0.048,0.047,0.049,0.048,0.04,0.049,0.048,0.04,0.04]
plt.plot(x,lbs_road,'-o',linewidth=3,label='AlexNet: LBS')
plt.plot(x,das_road,'--s',linewidth=3,label='AlexNet: DAS')
# for i in range(10):
#     plt.text(x[i],das_road[i]+0.04,das_road[i],ha='left',fontsize=14, va='bottom')
#     plt.quiver(x[i]+0.3, das_road[i]+0.04, -0.5, -1, color='black', scale=25, width=0.005)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(labelsize=16)
plt.ylabel('Average  Processing Time (s)',fontsize=18)
plt.xlabel('Length of the IFR Set',fontsize=18)
plt.legend(bbox_to_anchor=(0.3, 0.8),fontsize=20,ncol=1)
plt.grid(axis="y",linestyle='-.',zorder=0)
plt.show()

fig,ax=plt.subplots(figsize=(5,4))

width = 0.35
x=[1,2,3,4,5,6,7,8,9,10]
lbs = [random.uniform(15.5,16.1) for i in range(10)]
das_ax=[25.19,15.85,14.02,15.01,15.05,15.02,15.18,14.36,14.99,14.56]
error1=[0.049,0.048,0.047,0.049,0.048,0.04,0.049,0.048,0.04,0.04]
plt.plot(x,lbs,'-o',linewidth=3,label='VGG16: LBS')
plt.plot(x,das_ax,'--s',linewidth=3,label='VGG16: DAS')
# for i in range(10):
#     plt.text(x[i],das_ax[i]+0.8,das_ax[i],ha='left',fontsize=14, va='bottom')
#     plt.quiver(x[i]+0.3, das_ax[i]+0.04, -0.5, -1, color='black', scale=25, width=0.005)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(labelsize=16)
plt.ylabel('Average  Processing Time (s)',fontsize=18)
plt.xlabel('Length of the IFR Set',fontsize=18)
plt.legend(bbox_to_anchor=(0.3, 1),fontsize=20,ncol=1)
plt.grid(axis="y",linestyle='-.',zorder=0)
plt.show()