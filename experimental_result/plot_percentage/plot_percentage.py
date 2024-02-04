import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)

#条形图的绘制
labels = ['LBS-Road', 'DAS-Road', 'LBS-Parking', 'DAS-Parking']
plt.rc('font',family='Times New Roman')
def auto_text1(rects):
    for rect in rects:
        print(rect.get_x()+0.5)
        ax.text(rect.get_x()+0.5, rect.get_height()/2, rect.get_height(), ha='left',fontsize=13, va='bottom')
        plt.quiver(rect.get_x()+0.5, rect.get_height()/2+0.1, -1, -1, color='black', scale=25, width=0.005)

def auto_text2(rects1,rects2):
    for i  in range(len(rects2)):
        ax.text(rects1[i].get_x()+0.5, rects1[i].get_height()/2+rects2[i].get_height(), rects1[i].get_height(), ha='left',fontsize=13, va='bottom')
        plt.quiver(rects1[i].get_x()+0.5, rects1[i].get_height()/2+0.1+rects2[i].get_height(), -1, -1, color='black', scale=25, width=0.005)

def auto_text3(rects):
    for rect in rects:
        ax.text(rect.get_x()+0.5, rect.get_height()/2, rect.get_height(), ha='left',fontsize=13, va='bottom')
        plt.quiver(rect.get_x()+0.5, rect.get_height()/2+0.1, -1, -1, color='black', scale=25, width=0.005)

def auto_text4(rects):
    for rect in rects:
        ax.text(rect.get_x()+0.5, rect.get_height()/2, rect.get_height(), ha='left',fontsize=13, va='bottom')
        plt.quiver(rect.get_x()+0.5, rect.get_height()/2+0.1, -1, -1, color='black', scale=25, width=0.005)


ax_trans1=[1.194,0.184,1.191,0.123]
ax_inference1=[0.793,0.946,0.766,0.851]
ax_inference1=[round(i,3) for i in ax_inference1]
ax_trans2=[1.164,0.232,1.155,0.191]
ax_inference2=[0.782,0.686,0.761,0.566]


width = 0.35  # 条形图的宽度
fig,ax = plt.subplots(figsize=(6,5))
plt.tick_params(labelsize=15)
re1=ax.bar(labels, ax_trans1,width,color='bisque', hatch = '///',label=r'Master$\rightarrow$Work 0 ',zorder=10)
re2=ax.bar(labels,ax_inference1,width,bottom=ax_trans1,color='lightblue', hatch =  '*',label='Inference in Work 0',zorder=10)
re3=ax.bar(labels,ax_trans2,width,bottom=[ax_trans1[i]+ax_inference1[i] for i in range(4)],color='slateblue',hatch = '..',label=r'Work 0$\rightarrow$Work 1 ',zorder=10)
re4=ax.bar(labels,ax_inference2,width,bottom=[ax_trans1[i]+ax_inference1[i]+ax_trans2[i] for i in range(4)],color='orchid',hatch = '++',label='Inference in Work 1',zorder=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Scores')
ax.set_ylabel('Processing Time (s)',fontsize=15)
ax.legend(bbox_to_anchor=(1.1, 1.2),fontsize=15,ncol=2,shadow=False,framealpha=0)
plt.grid(axis="y",linestyle='-.',zorder=0)
x=[0.325,1.325,2.325,3.325]
y1=[i/2+0.1 for i in ax_trans1]
y2=[ax_trans1[i]+ax_inference1[i]/2+0.1 for i in range(4)]
y3=[ax_trans1[i]+ax_inference1[i]+ax_trans2[i]/2+0.1 for i in range(4)]
y4=[ax_trans1[i]+ax_inference1[i]+ax_trans2[i]+ax_inference2[i]/2+0.1 for i in range(4)]
y=[y1,y2,y3,y4]
for i in range(4):
    ax.text(x[i] , y1[i], ax_trans1[i], ha='left', fontsize=13, va='bottom')
    plt.quiver(x[i] , y1[i] + 0.1, -1, -1, color='black', scale=25, width=0.005)
    ax.text(x[i], y2[i], ax_inference1[i], ha='left', fontsize=13, va='bottom')
    plt.quiver(x[i], y2[i] + 0.1, -1, -1, color='black', scale=25, width=0.005)
    ax.text(x[i], y3[i], ax_trans2[i], ha='left', fontsize=13, va='bottom')
    plt.quiver(x[i], y3[i] + 0.1, -1, -1, color='black', scale=25, width=0.005)
    ax.text(x[i], y4[i], ax_inference2[i], ha='left', fontsize=13, va='bottom')
    plt.quiver(x[i], y4[i] + 0.1, -1, -1, color='black', scale=25, width=0.005)
plt.show()


labels = ['LBS-Road', 'DAS-Road', 'LBS-Parking', 'DAS-Parking']
vgg_trans1=[1.195,0.172,1.177,0.118]
vgg_inference1=[15.075,13.387,13.636,13.367]
vgg_trans2=[6.184,4.452,6.206,2.802]
vgg_inference2=[12.233,12.504,12.321,12.374]
fig,ax = plt.subplots(figsize=(6,5))
plt.tick_params(labelsize=15)
re1=ax.bar(labels, vgg_trans1,width,color='bisque', hatch = '///',label=r'Master$\rightarrow$Work 0 ',zorder=10)
re2=ax.bar(labels,vgg_inference1,width,bottom=vgg_trans1,color='lightblue', hatch =  '*',label='Inference in Work 0',zorder=10)
re3=ax.bar(labels,vgg_trans2,width,bottom=[vgg_trans1[i]+vgg_inference1[i] for i in range(4)],color='slateblue',hatch = '..',label=r'Work 0$\rightarrow$Work 1 ',zorder=10)
re4=ax.bar(labels,vgg_inference2,width,bottom=[vgg_trans1[i]+vgg_inference1[i]+vgg_trans2[i] for i in range(4)],color='orchid',hatch = '++',label='Inference in Work 1',zorder=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Scores')
ax.set_ylabel('Processing Time (s)',fontsize=15)
ax.legend(bbox_to_anchor=(1.1, 1.2),fontsize=15,ncol=2,shadow=False,framealpha=0)
plt.grid(axis="y",linestyle='-.',zorder=0)
x=[0.325,1.325,2.325,3.325]
y11=[i/2+1 for i in vgg_trans1]
y21=[vgg_trans1[i]+vgg_inference1[i]/2+1 for i in range(4)]
y31=[vgg_trans1[i]+vgg_inference1[i]+vgg_trans2[i]/2+1 for i in range(4)]
y41=[vgg_trans1[i]+vgg_inference1[i]+vgg_trans2[i]+vgg_inference2[i]/2+1 for i in range(4)]
y1=[y1,y2,y3,y4]
for i in range(4):
    ax.text(x[i] , y11[i]+0.2, vgg_trans1[i], ha='left', fontsize=13, va='bottom')
    plt.quiver(x[i] , y11[i] +0.5, -1, -1, color='black', scale=25, width=0.005)
    ax.text(x[i], y21[i], vgg_inference1[i], ha='left', fontsize=13, va='bottom')
    plt.quiver(x[i], y21[i] + 0.1, -1, -1, color='black', scale=25, width=0.005)
    ax.text(x[i], y31[i], vgg_trans2[i], ha='left', fontsize=13, va='bottom')
    plt.quiver(x[i], y31[i] + 0.1, -1, -1, color='black', scale=25, width=0.005)
    ax.text(x[i], y41[i], vgg_inference2[i], ha='left', fontsize=13, va='bottom')
    plt.quiver(x[i], y41[i] + 0.1, -1, -1, color='black', scale=25, width=0.005)
plt.show()