import numpy as np
import tensorflow as tf
import math
import random
import context
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

slim = tf.contrib.slim

seqdic={'A': [-1.043894014433722, -1.1744404390294072, -1.461785760426751, -0.17133173880554614, -0.06187410033338803, 1.5652288206303868, -0.9100202646805605 ],
'G': [-2.387468009883861, -1.663790621958327, -1.9910347685465015, -0.48012731455972774, -0.08500460513091643, -1.7480292668353965, -1.7924641577041347 ],
'V': [1.4648105551958346, -0.2936101097573518, -0.4032877441872501, 0.7351326932470514, -0.11391773612782732, -0.14852536254156973, 1.9579223876460548 ],
'L': [0.3311699965347796, 0.19574007317156777, 0.12596126393250037, 1.2132677782857844, -0.10235248372906286, 1.2224779839959958, -0.02757637165698671 ],
'I': [2.0106374908474542, 0.19574007317156777, 0.12596126393250037, 1.3128792543355203, -0.10235248372906286, 0.19422547409282134, 1.516700441134268 ],
'F': [0.6985535109156771, 1.1744404390294068, 1.1262418892788286, 1.3029181067305469, -0.31630965310620046, 0.19422547409282134, 0.7445620347386406 ],
'Y': [0.6985535109156771, 1.2723104756151908, 1.4332063139882838, 0.4761428555177379, -0.32209227930558243, -0.3770259202978309, 1.0754784946224805 ],
'W': [0.9819636505809408, 2.348880878058814, 2.2852972170610824, 1.7611308965593324, -0.16017874572288363, 0.4227260318490825, 1.1857839812504274 ],
'T': [0.7930235574707648, -0.5872202195147037, -0.6149873474351503, -0.22113747683041413, -0.3567880365018753, -0.8340270358103529, 0.523951061482747 ],
'S': [-1.0124039989153593, -1.0765704024436233, -1.1442363555549007, -0.5199719049796222, -0.29896177450805406, -0.9482773146884831, -0.3584928315408266 ],
'R': [0.06875320054842428, 1.1744404390294068, 1.2532616512275687, -1.4862032226620614, 2.615481829980522, 0.8797271473616041, -0.6894092914246671 ],
'K': [-0.4035970322270152, 0.4893501829289195, 0.533483000184708, -1.4662809274521142, 2.181784865026865, 0.4227260318490825, -0.4687983181687735 ],
'H': [0.7510368701129484, 0.5872202195147036, 0.4752656092915358, -0.35063239569507093, 0.8517808391689832, -0.14852536254156973, -0.13788185828493355 ],
'D': [-0.7080005155711872, -0.5872202195147037, -0.5197225259735953, -1.247135680142695, -1.8891839793381304, -0.3770259202978309, -1.2409367245644007 ],
'E': [-0.749987202929004, -0.19574007317156802, 0.009526482146155164, -1.1176407612780384, -1.8082272125467813, 1.5652288206303868, -1.1306312379364543 ],
'N': [-0.7080005155711872, -0.39148014634313577, -0.42975019459323754, -1.0777961708581438, 0.17521357384127748, -0.8340270358103529, -1.0203257513085073 ],
'Q': [-0.749987202929004, 0.09787003658578375, 0.09949881352651294, -0.6992725618691469, -0.3278749055049644, 0.8797271473616041, -0.6894092914246671 ],
'M': [0.07924987238787873, 0.4893501829289195, 0.35353833742399293, 0.7450938408520251, -0.2931791483086721, 1.1082277051178653, 0.08272911497096014 ],
'P': [0.41514337125041334, -1.663790621958327, -0.5514774664607801, 0.23707531299837145, 0.3371271074239763, -1.7480292668353965, 0.3033400882268538 ],
'C': [-0.5295570943004656, -0.39148014634313577, -0.7049596788155078, 1.0538894166062067, 0.07690892845178188, -1.2910281513228745, 1.0754784946224805 ],
'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]}

labeldic={'H':1,'G':1,'I':1, 'E':2,'B':2}


cnt=-1.754416150639155
std=3.0633068044491645
cnt_hmm=5521.74233031901
std_hmm=3322.2521438279464

data=np.load('cullpdb_test.npy').item()
seq=data['seq']
pssm_list=data['pssm']
dssp=data['dssp']
hmm_list=data['hhm']

print(len(seq))

for i in range(len(pssm_list)):
	pssm_list[i]=pssm_list[i]-cnt
	pssm_list[i]=pssm_list[i]/std

for i in range(len(hmm_list)):
	hmm_list[i]=hmm_list[i]-cnt_hmm
	hmm_list[i]=hmm_list[i]/std_hmm


def g_data_test(n):
	l=len(seq[n])
	data1=np.zeros([l,20])
	data2=np.zeros([l,30])
	data3=np.zeros([l,7])
	label=np.zeros([l,3])
	z=np.zeros([l])
	for i in range(len(seq[n])):
		if dssp[n][i] in labeldic:
			label[i][int(labeldic[dssp[n][i]])]=1.0
		else:
			label[i][0]=1.0
		data1[i]=pssm_list[n][i]
		data2[i]=hmm_list[n][i]
		if seq[n][i] in seqdic:
			data3[i]=seqdic[seq[n][i]]
		else:
			data3[i]=seqdic['X']
		z[i]=1.0
	data=np.concatenate((data1,data2,data3),axis=1)
	data=data.reshape([1,l,1,57])
	label=label.reshape([1,l,3])
	z=z.reshape([1,l])
	return data,label,z



learning_rate = 0.04
training_epochs = 40
batch_size = 5


x=tf.placeholder("float",[None,None,1,57],name='input')
y=tf.placeholder("float",[None,None,3],name='label')
z=tf.placeholder("float",[None,None],name='z')
lr=tf.placeholder("float",[],name='lr')
net=context.cnn_context(x)

net_rsp=tf.reshape(net,[-1,3])
y_rsp=tf.reshape(y,[-1,3])
z_rsp=tf.reshape(z,[-1])
net_soft=tf.nn.softmax(net_rsp)
pre=tf.argmax(net_rsp,1)

c_e=tf.nn.softmax_cross_entropy_with_logits(logits=net_rsp, labels=y_rsp)
tt1=tf.reduce_sum(c_e*z_rsp)
tt2=tf.reduce_sum(z_rsp)
loss=tt1/tt2

accuracy_ = tf.cast(tf.equal(tf.argmax(net_rsp,1),tf.argmax(y_rsp,1)),tf.float32)
ac_num = tf.reduce_sum(accuracy_*z_rsp)
accuracy = ac_num/tt2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)


init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=0)

with tf.Session() as sess:
	all_data=[]
	all_pre=[]
	all_r=[]
	for cw in range(10):
		save_name="model/my-model-"+str(cw+1)
		acc=0.0
		sess.run(init)
		saver.restore(sess,save_name)
		t1=0.0
		t2=0.0
		eve_ac=[]
		all_pret=[]
		for i in range(len(seq)):
			batch_x,batch_y,batch_z=g_data_test(i)
			n1,n2,b_pre=sess.run([ac_num,tt2,pre],feed_dict={x:batch_x, y:batch_y, z:batch_z})
			t1=t1+n1
			t2=t2+n2
			eve_ac.append(n1/n2)
			all_pret.append(b_pre)
		print("acc:",t1/t2)
		all_r.append(t1/t2)
		all_data.append(eve_ac)
		all_pre.append(all_pret)


t1=0
t2=0
for i in range(len(seq)):
	d=[]
	for j in range(len(all_pre)):
		d.append(all_pre[j][i])
	p=[]
	l=len(seq[i])
	for j in range(l):
		ss=[0,0,0]
		for k in range(len(d)):
			ss[int(d[k][j])]+=1
		p.append(np.argmax(ss))
	for j in range(len(p)):
		if dssp[i][j] in labeldic:
			dssp_num=labeldic[dssp[i][j]]
		else:
			dssp_num=0
		if p[j]==dssp_num:
			t1+=1
		t2+=1

reve=np.mean(all_r)
rstd=np.std(all_r)

print('eveacc:',reve)
print('accstd:',rstd)
print('voteacc:',t1/t2)
