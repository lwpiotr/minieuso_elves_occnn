#!/usr/bin/python

import numpy as np
import torch
from etoshelpers import *
import pickle
import uproot
import glob
import sys
import os
from etoshelpers import *
import ROOT

#mymlp = CNN()


#xp=cp
nl = 128
ncl = 128*2
ncl=16

norm = 1
anscombe=True
freemantukey=False
median=False

def main():

	file_list = sorted(glob.glob("CPU*root"))
#	print(file_list)
#	exit()

	# Init the neural network

	from train_elves_occnn import CNN

	elves_nn = CNN(nl, ncl)
	if len(sys.argv)>1:
		a = torch.load(sys.argv[1])
		elves_nn.load_state_dict(torch.load(sys.argv[1]))
	else:
		elves_nn.load_state_dict(torch.load('/home/lewhoo/workspace/minieuso_elves_cnn/pytorch/elves_samples.pk/res_batchsize32/curve_snapshot_epoch90.model'))
	elves_nn.eval()
	device="cuda"
	elves_nn.to(device)
	

	#a = serializers.load_npz('81_100_0vs100_6cnn_largegap_single8bank/curve.model', elves_nn)
#	a = serializers.load_npz('occnn_bg99.86_elf1/curve.model', elves_nn)
#	a = serializers.load_npz('occnn_bg99.86_elf1/curve.model', elves_nn)
#	a = serializers.load_npz('/home/lewhoo/workspace/minieuso_elves_cnn/occnn_bg99.84_elf1_better/curve.model', elves_nn)
	#a = serializers.load_npz('curve.model', elves_nn)
#	device = chainer.get_device(0)
#	elves_nn.to_device(0)
#	device.use()

#	elves_nn.centre = chainer.Variable(cp.loadtxt("cur_centre.txt"))

	flat = None
	if os.path.exists("pdm.npy"):
		flat = np.load("pdm.npy")
		flat[flat<0.005]=1
		flat = np.fliplr(np.rot90(flat, 3))

	packets_count = 0

	for fn in file_list:
#		print(fn)
		# Load the data from file to memory
		try:
			t = uproot.open(f"{fn}:tevent")
		except:
			print(f"FAILED TO OPEN {fn}")
			continue
			

		packets_count+=t.num_entries/128
			
		pc = t["photon_count_data"].array().to_numpy().astype(np.float32)[:,0,0]
		if flat is not None:
			pc/=flat
		pc = torch.as_tensor(pc).to(device)
#		print(pc.shape)
		# Reshape to have every packet separately and add dimension for channel
		pc = pc.reshape((pc.shape[0]//128,1,128,48,48))
		
		# No events in the file, continue to the next
		if pc.shape[0]==0:
			print(f"No events in {fn}")
			continue
		
#		print(fn, pc.shape)
		# Normalize each packet
	#	mean = cp.mean(pc, axis=(1,2,3,4)).reshape(pc.shape[0], 1, 1, 1, 1)
	#	std = cp.std(pc, axis=(1,2,3,4)).reshape(pc.shape[0], 1, 1, 1, 1)
	#	print(mean.shape, std.shape)
	#	pc = (pc-mean)/std

		#pc[pc>255]=255
		
		for i,el in enumerate(pc):
			if norm==1:
				el[el>1000]=1000
				el[el<0]=0
				if anscombe: el = 2*torch.sqrt(el+3/8)
				elif freemantukey: el = torch.sqrt(el+1)+torch.sqrt(el)
				pmeans = torch.mean(el, axis=(0,1))
				pstds = torch.std(el, axis=(0,1))
				pstds[pstds==0]=1		
				#print(pmeans.shape)
				el1 = (el-pmeans)/pstds
			elif norm==2:
				el[el>1000]=1000
				el[el<0]=0
				mean = torch.mean(el)
				std = torch.std(el)
				el1 = (el-mean)/std
				el1/=torch.max(el1)		
			elif norm==3:
				el[el>1000]=1000
				el[el<0]=0
				mean = torch.mean(el)
				#std = torch.std(el)
				el1 = (el-mean)
				el1/=torch.max(el1)
			elif norm==4:
				el[el>1000]=1000
				el[el<0]=0			
				mx = torch.max(el)
				el1=el-mx/2
				el1/=mx/2		
			elif norm==5:
				el[el>1000]=1000
				el[el<0]=0			
				mx = torch.max(el)
				el1=el/mx
			elif norm==6:
				el[el>1000]=1000
				el[el<0]=0			
				if anscombe: el = 2*torch.sqrt(el+3/8)
				elif freemantukey: el = torch.sqrt(el+1)+torch.sqrt(el)
				pmeans = torch.mean(el, axis=(0,1))
				pstds = torch.std(el, axis=(0,1))
				pstds[pstds==0]=1
				#print(pmeans.shape)
				el1 = (el-pmeans)/pstds
				el1 /= torch.max(el1)
			elif norm==7:
				el[el>1000]=1000
				el[el<0]=0			
				if median: pmeans = torch.as_tensor(np.median(el.to("cpu").data, axis=(0,1))).to("cuda")
				else: pmeans = torch.mean(el, axis=(0,1))
				pstds = torch.std(el, axis=(0,1))
				pstds[pstds==0]=1
				#print(pmeans.shape)
				el1 = (el-pmeans)#/pstds
				el1 /= torch.max(el1)
			elif norm==8:
				el[el>1000]=1000
				el[el<0]=0			
				pmeans = torch.mean(el, axis=(0,1))
				pstds = torch.std(el, axis=(0,1))
				pstds[pstds==0]=1
				#print(pmeans.shape)
				el1 = (el-pmeans)#/pstds
				el1 /= 1000
			elif norm==9:
				el[el>1000]=1000
				el[el<0]=0			
				pmeans = torch.mean(el, axis=(0,1))
				pstds = torch.std(el, axis=(0,1))
				pstds[pstds==0]=1
				#print(pmeans.shape)
				el1 = (el-pmeans)#/pstds
			elif norm==10:
				el[el>1000]=1000
				el[el<0]=0			
				pmeans = torch.as_tensor(np.median(el.to("cpu").data, axis=(0,1))).to("cuda")
				pstds = torch.std(el, axis=(0,1))
				pstds[pstds==0]=1
				#print(pmeans.shape)
				el1 = (el-pmeans)/pstds
				el1 /= torch.max(el1)
				
			
			pc[i]=el1

#		print(pc.shape)

		# Loop through packet batches
		y = []
		dists = []
#		print(pc.shape[0]//10+1)
		for batch in range(pc.shape[0]//10+1):
#			print("in loop")
			if batch*10>pc.shape[0]-1:
#				print("breaking")
				break
			# Feed the NN model
			labels = torch.ones((batch,1))
#			print(elves_nn.predict(pc[batch*10:(batch+1)*10], labels))
			#exit()
			with torch.no_grad():
				res, dist = elves_nn.predict_dist(pc[batch*10:(batch+1)*10], labels)
			y.append(res)
			dists.append(dist)
#			print(batch, y, batch*10, (batch+1)*10, pc[batch*10:(batch+1)*10].shape)

		#print(y)

		y = torch.hstack(y)
#		print(y)
		dists = torch.hstack(dists)

		if len(sys.argv)>2:
			out_name = sys.argv[2]
		else:
			out_name = "tle_occnn_results.txt"
		
		# If there are some events, open the file with ROOT to filter out the sunlight - do not know how to access this leaf with uproot :(
		if len(y)>0:
			rf = ROOT.TFile(fn, "read")
			rt = rf.Get("tevent")			
		
		with open(out_name, "a") as fl:
			for i,el in enumerate(y):
				if el:
					#print(fn, i*128, dists[i].item())
					add = True
					try:
						rt.GetEntry(i*128)
						if rt.Sun_from_ISS_alt>=-20.8: add=False
					except:
						pass
						
					
					if add: print(fn, i*128, dists[i].item(), file=fl)
	exit()


		
if __name__ == '__main__':
	a = main()


