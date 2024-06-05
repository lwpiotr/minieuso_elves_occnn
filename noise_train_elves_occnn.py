#!/usr/bin/python
# As standard train elves, but auguments data with different levels of noise and try to show results per noise level

# Based on https://www.sciencedirect.com/science/article/pii/S1877050917318343

import sys
import pickle
import time
import argparse
import os
import numpy as np
#from tqdm import tqdm
	
import torch
import torch.nn.functional as L
import torch.nn.functional as F
	
from etoshelpers import arrays2canvas, wait4key, pad_refresh, arrays2graph, create_fill_canvas_with_histogram_1D, arrays2multigraph, root_colors, array2canvas
import ROOT
from collections import defaultdict
############################################################

#torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

mod_name = vars(sys.modules[__name__])['__package__']

gmargin = 2

# If run as a script
if mod_name is None:

	parser = argparse.ArgumentParser(description='Chainer example: MNIST')
	parser.add_argument('--batchsize', '-b', type=int, default=3,
			help='Number of images in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=5,
			help='Number of sweeps over the dataset to train')
	parser.add_argument('--frequency', '-f', type=int, default=-1,
			help='Frequency of taking a snapshot')
	parser.add_argument('--device', '-d', type=str, default='cuda',
			help='Device specifier. Either ChainerX device '
			'specifier or an integer. If non-negative integer, '
			'CuPy arrays with specified device id are used. If '
			'negative integer, NumPy arrays are used')
	parser.add_argument('--out', '-o', default='result',
			help='Directory to output the result')
	parser.add_argument('--resume', '-r', type=str,
			help='Resume the training from snapshot')
	parser.add_argument('--autoload', action='store_true',
			help='Automatically load trainer snapshots in case'
			' of preemption or other temporary system failure')
	parser.add_argument('--unit', '-u', type=int, default=4000,
			help='Number of units')
	group = parser.add_argument_group('deprecated arguments')
	group.add_argument('--gpu', '-g', dest='device',
			   type=int, nargs='?', const=0,
			   help='GPU ID (negative value indicates CPU)')
	args = parser.parse_args()

	#chainer.cuda.set_max_workspace_size(256 * 1024 * 1024)

	device = torch.device(args.device)

	print('Device: {}'.format(device))
	print('# unit: {}'.format(args.unit))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('')

	#F.Convolution2D = L.Convolution2D

#nl = 1152
nl = 128
ncl = 128*2
ncl=16
#ncl1 = 1152//2
#ncl = 4608

#F.relu = F.leaky_relu
#F.relu = F.swish

# Model without MLP part for predicting tracks
class CNN(torch.nn.Module):
	def __init__(self, nl, ncl, device="cuda"):
		super(CNN, self).__init__()
		
		padding = 1
		
		self.device = device
		
		#with self.init_scope():
		# ToDo: Try with these bias and weight in PyTorch, or not if the default gives good results
		#self.bn0 = torch.nn.LazyInstanceNorm3d(affine=True, momentum=0.9)
		self.conv1=torch.nn.LazyConv3d(32, 3, padding=padding)#, initialW=initializers.HeNormal(), initial_bias=0.01)
		#self.bn1 = torch.nn.LazyBatchNorm3d(momentum=0.9)
		self.bn1 = torch.nn.Dropout(0.2)
		self.conv2=torch.nn.Conv3d(32, 32, 3, padding=padding)#, initialW=initializers.HeNormal(), initial_bias=0.01)
		self.bn2 = torch.nn.Dropout(0.2)
		self.conv3=torch.nn.Conv3d(32, 32, 3, padding=padding)#, initialW=initializers.HeNormal(), initial_bias=0.01)
		self.bn3 = torch.nn.Dropout(0.2)
		self.conv4=torch.nn.Conv3d(32, 32, 3, padding=padding)#, initialW=initializers.HeNormal(), initial_bias=0.01)
		self.bn4 = torch.nn.Dropout(0.2)
		self.conv5=torch.nn.Conv3d(32, 32, 3, padding=1)#, initialW=initializers.HeNormal(), initial_bias=0.01)
		self.bn5 = torch.nn.Dropout(0.2)
		self.conv6=torch.nn.Conv3d(32, 32, 3, padding=1)
		self.bn6 = torch.nn.Dropout(0.2)
		#self.conv7=torch.nn.Conv3d(32, 32, 3, padding=1)
		#self.lc1=L.Linear(None, ncl, initialW=initializers.LeCunUniform(), initial_bias=5e-5)
		self.lc1=torch.nn.LazyLinear(nl)
		#self.bn7 = torch.nn.LazyInstanceNorm3d(affine=True, momentum=0.9)
		#self.lc15=torch.nn.Linear(nl, nl)
		self.lc2=torch.nn.Linear(nl, ncl)
		#self.l2=torch.nn.LazyLinear(1, initialW=initializers.LeCunUniform(), initial_bias=0.5)
		
		"""
		self.bn0=L.BatchNormalization((1,128,48,48), decay=0.9, eps=0.001)
		self.bn1=L.BatchNormalization(32, decay=0.9, eps=0.001)
		self.bn2=L.BatchNormalization(32, decay=0.9, eps=0.001)
		self.bn3=L.BatchNormalization(32, decay=0.9, eps=0.001)
		self.sw1 = L.Swish(None)
		self.sw2 = L.Swish(None)
		self.sw3 = L.Swish(None)
		self.sw4 = L.Swish(None)
		self.sw5 = L.Swish(None)
		"""
		#bn4=L.BatchNormalization(ncl, decay=0.9, eps=0.001)
		
		#self.centre = chainer.Parameter(xp.zeros((1,ncl), dtype=xp.float32), update_rule=False)
		# ToDo: Convert to some PyTorch Variable equivalent?
		self.centre = torch.nn.Parameter(torch.zeros(ncl), requires_grad=True)
		#self.centre = xp.array([0], dtype=xp.float32)

	def init_wb(self):
		torch.nn.init.kaiming_normal_(self.conv1.weight)
		self.conv1.bias.data.fill_(0.01)
		torch.nn.init.kaiming_normal_(self.conv2.weight)
		self.conv2.bias.data.fill_(0.01)
		torch.nn.init.kaiming_normal_(self.conv3.weight)
		self.conv3.bias.data.fill_(0.01)
		torch.nn.init.kaiming_normal_(self.conv4.weight)
		self.conv4.bias.data.fill_(0.01)
		torch.nn.init.kaiming_normal_(self.conv5.weight)
		self.conv5.bias.data.fill_(0.01)
		self.centre.data = torch.zeros(ncl).to(device)
		
		#torch.nn.init.xavier_normal_(self.lc1.weight)
		#torch.nn.init.xavier_normal_(self.lc2.weight)
	

	def forward_model(self, x):
		max_pool_kernel = (2,2,2)
		ceil_mode = True
		#hc = F.relu(self.conv1(x))
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("conv1")
		#hc = F.max_pool3d(F.relu(self.conv1(F.dropout(x, 0.2, training=self.training))), max_pool_kernel, ceil_mode=ceil_mode)
		hc = F.max_pool3d(F.relu(self.bn1(self.conv1(x))), max_pool_kernel, ceil_mode=ceil_mode)
		#print(hc.shape)
		#exit()
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("mp1")

		#hc = self.bn1(hc)

		#hc = F.relu(self.conv2(hc))
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("conv2")
		
		#hc = F.max_pool3d(F.relu(self.conv2(hc)), max_pool_kernel,  ceil_mode=ceil_mode)
		hc = F.max_pool3d(F.relu(self.bn2(self.conv2(hc))), max_pool_kernel,  ceil_mode=ceil_mode)
		#exit()		
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("mp2")
		
		#hc = self.bn2(hc)
		
		#hc = F.relu(self.conv3(hc))
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("conv3")
		
		#hc = F.max_pool3d(F.relu(self.conv3(hc)), max_pool_kernel,  ceil_mode=ceil_mode)
		hc = F.max_pool3d(F.relu(self.bn3(self.conv3(hc))), max_pool_kernel,  ceil_mode=ceil_mode)
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("mp3")
		
		#hc = F.relu(self.conv4(hc))
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("conv4")
		
		#hc = F.max_pool3d(F.relu(self.conv4(hc)), max_pool_kernel,  ceil_mode=ceil_mode)
		hc = F.max_pool3d(F.relu(self.bn4(self.conv4(hc))), max_pool_kernel,  ceil_mode=ceil_mode)
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("mp4")
		
		#hc = F.relu(self.conv5(hc))
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("conv5")
		
		#hc = F.max_pool3d(F.relu(self.conv5(hc)), max_pool_kernel,  ceil_mode=ceil_mode)
		hc = F.max_pool3d(F.relu(self.bn5(self.conv5(hc))), max_pool_kernel,  ceil_mode=ceil_mode)
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("mp5")
		
		#hc = F.relu(self.conv6(hc))
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("conv6")
		
		#hc = F.max_pool3d(F.relu(self.conv6(hc)), max_pool_kernel,  ceil_mode=ceil_mode)
		hc = F.max_pool3d(F.relu(self.bn6(self.conv6(hc))), max_pool_kernel,  ceil_mode=ceil_mode)
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("mp6")
		

		#hc = F.relu(self.conv7(hc))
		#hc = F.max_pool3d(hc,2, ceil_mode=True)
		
		#print("hc", hc.shape)
		#hc = hc.resize(hc.shape[0], np.prod(hc.shape[1:]))
		#hc = self.bn7(hc)
		hc = hc.view(hc.shape[0], np.prod(hc.shape[1:]))
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("view")
		
		#print("hc", hc.shape)		
		#hc = F.max_pool3d(F.relu(self.conv3(hc)),2)
		# If I use a layer nn.dropout, it will automatically set training/eval and appear in some summaries
		hc = F.dropout(self.lc1(hc), 0.5, training=self.training)
		#if torch.any(torch.isnan(hc)) or torch.any(torch.isinf(hc)):
		#	print("lc1")
		
	
		#hc = F.dropout(self.lc15(hc), 0.5, training=self.training)
		#print("hc", hc.shape)
		#hc = self.lc2(hc)
		#if torch.any(torch.isnan(hc0)) or torch.any(torch.isinf(hc0)):
		#	print("llll")
		#	print("oho", torch.any(hc0>100))
		
		#hc = F.dropout(self.lc2(hc), 0.5, training=self.training)
		hc = self.lc2(hc)
		#if torch.any(torch.isnan(hc1)) or torch.any(torch.isinf(hc1)):
		#	print("lc2")
		#	print("oho1", torch.any(hc>100), torch.max(hc), torch.max(self.lc1.weight), torch.max(self.lc1.bias))
		#	print("oho2", torch.any(hc1>100), torch.max(hc1), torch.max(self.lc2.weight), torch.max(self.lc2.bias))
		#	print("oho3", torch.any(hc0>100), torch.max(hc0))
		
		#print("hc", hc.shape)
		#return hc/500
		return hc*2

	# X is the data, y is the label
	def forward(self, x, lab):
		# Compute the layers on the data
		h = self.forward_model(x)
		
		#if torch.any(torch.isnan(h)) or torch.any(torch.isinf(h)):
		#	print("forward", h, x)
		#	print("ff", h.shape, x.shape)
		#	print(torch.isnan(x).shape, torch.isinf(x).shape)
		#	print("f1", torch.any(torch.isnan(x)), torch.any(torch.isinf(x)))
		#	print("f2", torch.any(x>100), torch.any((x>-1e-5) & (x<1e-5)))

		#lab = xp.array(lab[..., None])
		#harr = h.array
		#print(x.shape, h.shape, lab.shape)

		# Compute the centre of the positive (y=1) samples
		# If there are any positive samples in the batch
		"""
		if np.any(lab==1):
			#self.centre = chainer.Variable((xp.sum((1-lab)*harr+lab*harr)/xp.sum(lab)).astype(xp.float32))
			# This is wrong because:
			# 1. It gives a scalar, while centre is the vector in the hyperspace
			# 2. There is probably a mistake in the paper - the below calculates the centre for all the samples, not only for non-defective ones
			#self.centre = chainer.Variable(xp.array([(xp.sum((1-lab)*harr+lab*harr)/xp.sum(lab)).astype(xp.float32)]))
			# This should fix both the issues below
			#print(lab.shape, harr.shape)
			#self.centre = chainer.Variable(xp.array([(xp.sum(lab*harr, axis=0)/xp.sum(lab)).astype(xp.float32)]))
			
			# ToDo: Convert to some PyTorch Variable equivalent?
			#self.centre = chainer.Variable(xp.array([0], dtype=xp.float32))
			self.centre = torch.nn.Parameter(torch.zeros(1))
		"""
		#self.centre = torch.nn.Parameter(torch.zeros(h.shape)).to(self.device)
		if torch.any(lab==1) and self.training:
			pass
			#print(h.shape)
			#self.centre.data = torch.sum(h[lab==1], axis=0)/torch.sum(lab)
			#print("hh", h)
			#print("centre", self.centre, torch.sum(lab), h[:,0], torch.sum(h[lab==1][:,0]), torch.sum(h[lab==1][:,0])/torch.sum(lab), torch.sum(h[lab==1], axis=0))
			#if torch.sum(lab)>1: exit()
			
			
			#self.centre = chainer.Variable([0])
		# Broadcasting the centre to the number of labels (probably need the same centre, but given separately for each sample) - if not done always, then the initial case with all 0 is not reshaped properly. Should be done better ;)
		#if chainer.config.train:
		#self.centre = self.centre[0]*torch.ones(h.shape)
#			print("train", self.centre, lab)
		#print("Centre", self.centre, self.centre.shape)
		
		# Compute the distanance of the samples from the centre
		#distance = chainer.Variable(xp.linalg.norm(harr-self.centre.array))
		
		# Return the standard CNN output and the distance needed for contrastive loss
		#print("centre forward", self.centre)
		#print(h, self.centre)
		return h, self.centre
		#return h
		
	def predict(self, x, lab):
		#print("self", self.centre)
		h, centre = self.forward(x, lab)
		#print(h.shape, centre.shape)
		diff = h - centre
		dist_sq = torch.sum(diff ** 2, axis=1)
		dist = torch.sqrt(dist_sq)
		margin=gmargin
		#print(dist, centre, margin)
		#return lab*(dist<=margin)+(1-lab)*(dist>margin)
		return dist<=margin

	def predict_dist(self, x, lab):
		#print("self", self.centre)
		h, centre = self.forward(x, lab)
		#print(h.shape, centre.shape)
		diff = h - centre
		dist_sq = torch.sum(diff ** 2, axis=1)
		dist = torch.sqrt(dist_sq)
		margin=gmargin
		#print("predict dist", margin, dist, dist<=margin)
		#print(dist, centre, margin)
		#return lab*(dist<=margin)+(1-lab)*(dist>margin)
		return dist<=margin, dist

	def loss_predict(self, x, lab):
		#centre = self.centre[0]
		#print("calling forward")
		h, centre = self.forward(x, lab)
		#print(x, h)
		#centre*=xp.ones(shape=h.shape)
		#centre = torch.nn.Parameter(torch.zeros(h.shape)).to(self.device)
		diff = h - centre
		dist_sq = torch.sum(diff ** 2, axis=1)
		dist = torch.sqrt(dist_sq)
		margin=gmargin
		#print("predict", margin)
		#print("h", h, h.shape)
		#print("centre", centre)
		#exit()
#		print("tu", centre, margin, dist, lab, dist<=margin, xp.mean(((dist<=margin).astype(xp.float))==lab))
		#print("dist", dist, h)
		return h, centre, dist<=margin
		#print("losspr", margin, dist, lab, lab*(dist<=margin)+(1-lab)*(dist>margin))
		#exit()
		#return h, centre, lab*(dist<=margin)+(1-lab)*(dist>margin)
	

def main():

	scaler = torch.cuda.amp.GradScaler()

	mymlp = CNN(nl, ncl, device=args.device)
	model = mymlp
	mymlp.load_state_dict(torch.load('/home/lewhoo/workspace/minieuso_elves_cnn/pytorch/tuning/res_batchsize16_fixed_centre/curve_snapshot_epoch5.model'))
	model.to(device=device)
	
	import pickle
	#with open("elves_samples.pk", "rb") as f:
	with open("all_elves_samples.pk", "rb") as f:
		samples = pickle.load(f)

	samples1 = []
	cube, label = [], []
	c=0
	for eel in samples:
		el, ell = eel
		#print(ell[1], ell)
		#if ell[1]!="0": continue
		el[el>1000]=1000
		el[el<0]=0
		el = 2*np.sqrt(el+3/8)
		#el = np.sqrt(el+1)+np.sqrt(el)
		#"""
		pmeans = np.mean(el, axis=(0,1))
		#pmeans = np.median(el, axis=(0,1))
		pstds = np.std(el, axis=(0,1))
		pstds[pstds==0]=1
		#print(pmeans.shape)
		el1 = (el-pmeans)/pstds
		#c1 = array2canvas(el1[0,70])
		#wait4key()
		#el1 = (el-pmeans)#/pstds
		#el1/=np.max(el1)
		#el1 = (np.tanh(((el - pmeans) / pstds)) + 1)-0.5
		#el1/=1000
		#el1 = el1.astype(np.float16)
		#print(el1.shape)
		#print(el[0,:,0,0], el1[0,:,0,0])
		#samples1.append([el1, ell])
		#cube.append(torch.tensor(el1))
		#label.append(torch.tensor(ell))
		#"""
		#mean = np.mean(el)
		#if mean>1000 or mean<-1000: 
		#	print(el, el.shape, mean, np.min(el), np.max(el), np.any(np.isinf(el)), np.any(np.isnan(el)), ell)
		#	print(np.where(el<-1e10))
		#	#exit()
		#print(mean, std)
		#el1 = (el-mean)#/std
		#print(np.max(el1))
		#el1/=np.max(el1).astype(np.float16)
		#mx = np.max(el)
		#el1=el-mx/2
		#el1=el/mx
		#samples1.append([el1, ell])
		samples1.append([el1, ell])
		label.append(ell[0])
		#if np.any(np.isinf(el1)) or np.any(np.isnan(el1)):
		#	exit()
		#print(el1.shape, ell[0], len(ell), ell)
		
	samples = samples1
	print(len(samples))

	elves_count = np.count_nonzero(label)
	bg_count = np.count_nonzero(np.array(label)==0)
	print("elve and bg counts", elves_count, bg_count)

	
#	# Remove too high values from samples
#	pics = np.array([el[0] for el in samples])
#	pics[pics>255]=255
#	pmean = np.mean(pics, axis=(1,2,3,4)).reshape(pics.shape[0], 1, 1, 1, 1)
#	pstd = np.std(pics, axis=(1,2,3,4)).reshape(pics.shape[0], 1, 1, 1, 1)
#	pics = (pics-pmean)/pstd
#	samples = [[np.array(pics[i]), el[1]] for i,el in enumerate(samples)]

	#print(len(samples[0]), len(samples), samples[0][0].shape, "oo")
	#s = samples[0][0]
	#print(np.mean(s), np.any(s>355), s[np.where(s>355)])
	#exit()

	# Make form (elf, label) ((elf, label), label))
	#samples = [(el[0], el[1], el[1]) for el in samples]
	#samples = [(el[0], el[1], el[1]) for el in samples[:100]]
	#samples = [(el[0], el[1], el[1]) for el in samples]

	break_point = int(len(samples)*0.7)

	#train = samples[:1650]
	#test = samples[1650:]
	#train = samples[:break_point]
	#test = samples[break_point:]
	
	#print(len(samples), len(train), len(test))
	#exit()

	#print(train[0])
	#exit()

	max_accuracy = 0
	
	#train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	#test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
	
	#train = torch.utils.data.TensorDataset(cube[:break_point], label[:break_point])
	#test = torch.utils.data.TensorDataset(cube[break_point:], label[break_point:])
	
	sampler_train = torch.utils.data.WeightedRandomSampler(torch.tensor([1/elves_count if el==1 else 1/bg_count for el in label[:break_point]]), len(label))
	sampler_test = torch.utils.data.WeightedRandomSampler(torch.tensor([1/elves_count if el==1 else 1/bg_count for el in label[break_point:]]), len(label))
	
	#print(torch.tensor([1/elves_count if el==1 else 1/bg_count for el in label[:break_point]]), torch.tensor([1/elves_count if el==1 else 1/bg_count for el in label[break_point:]]))
		
	train_iter = torch.utils.data.DataLoader(samples[:break_point], batch_size=args.batchsize, num_workers=4, sampler=sampler_train)
	test_iter = torch.utils.data.DataLoader(samples[break_point:], batch_size=args.batchsize, num_workers=4, sampler=sampler_test)
	#train_iter = torch.utils.data.DataLoader(samples[:break_point], batch_size=args.batchsize, num_workers=4, shuffle=True)
	#test_iter = torch.utils.data.DataLoader(samples[break_point:], batch_size=args.batchsize, num_workers=4, shuffle=False)	
	#train_iter = DataLoader(train, batch_size=args.batchsize)
	#test_iter = DataLoader(test, batch_size=args.batchsize, repeat=False, shuffle=False)
	
	print(len(train_iter), len(samples[:break_point]), len(samples[break_point:]))

	"""
	# Forward the model to init the lazy layers
	with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
		with torch.no_grad():
			model(torch.ones((1,)+samples[0][0].shape, dtype=torch.float16).to(device), torch.tensor(1).to(device))
		
			# Init the model weights/biases after the first pass
			model.init_wb()
	"""
	
	#optimizer = torch.optim.AdamW(model.parameters())
	optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-5)
	#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.5, cycle_momentum=False)
	#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, cycle_momentum=False, steps_per_epoch=len(train_iter), epochs=args.epoch, max_momentum=0.99)#, epochs=100, steps_per_epoch=10)

	print(model)


	contrastive_loss = ContrastiveLoss()
	
	#"""

	cl = ROOT.TCanvas("loss", "loss")
	ca = ROOT.TCanvas("acc", "acc")
	gtl = ROOT.TGraph()
	gvl = ROOT.TGraph()
	gta = ROOT.TGraph()
	ga = ROOT.TGraph()


	cltbg = ROOT.TCanvas()
	clvbg = ROOT.TCanvas()
	ctabg = ROOT.TCanvas()
	cabg = ROOT.TCanvas()

	#"""

	#timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	#writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
	#epoch_number = 0

	#lr=1e-8
	#mult = (1/1e-8)**(1/(len(train_iter)-1))
	#beta=0.98
	#avg_loss=0
	#print(mult)
	#exit()
	#optimizer.param_groups[0]["lr"]=lr
	#lrs, losses = [], []

	gtrain_loss_bg, gtest_loss_bg, gtrain_acc_bg, gtest_acc_bg = defaultdict(ROOT.TGraph), defaultdict(ROOT.TGraph), defaultdict(ROOT.TGraph), defaultdict(ROOT.TGraph)

	#while train_iter.epoch < args.epoch:
	for epoch_index in range(args.epoch):
		print("Epoch", epoch_index)
		
		model.train(True)
		
		running_loss = 0.
		last_loss = 0.		
		
		train_loss_bg, test_loss_bg, train_acc_bg, test_acc_bg = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
		loss_list_train, loss_list_test, acc_list_test = [], [], []
		bg_levels_list, losses_list = [], []
		test_losses = []
		test_accuracies = []
		train_accuracies = []
		
		
		# Iterate 8 times, to get each affine transform
		# ToDo: at the moment a sample can repeat :(
		for trans_num in range(8):
			# Iterate through batches
			for i, data in enumerate(train_iter):
				#print(i, trans_num)
				#train_batch = train_iter.next()
				#image_train, target_train = data
				image_train_orig, target_train_info = data


				# Randomly select affine transform for each sample
				for imin in range(len(image_train_orig)):
					random_affine_transform(image_train_orig[imin])
				#exit()

				#print("pre to device")
				#print("aaaa", target_train_info, target_train_info[0], image_train.shape, target_train_info[0].shape)
				#image_train, target_train = image_train.to(device), target_train.to(device)
				#print("post to device")
				image_train, target_train = image_train_orig.to(device), target_train_info[0].to(device)
				#print(i, len(image_train), len(target_train))
				#print(image_train[0].shape, target_train[0])
				#image_train, target_train, target_train = chainer.dataset.concat_examples(train_batch, device)
				#print(image_train.shape)
				#print(image_train.shape, target_train.shape, type(image_train), type(target_train))
				#exit()
				
				
				
				optimizer.zero_grad()
				
				#print("autocast")
				with torch.amp.autocast(device_type="cuda", dtype=torch.float16):

					# Calculate the prediction of the network
					#prediction_train, centre = model(image_train, target_train)
					prediction_train, centre, prediction_train_classified = model.loss_predict(image_train, target_train)
					#print("devices", prediction_train.device, centre.device, target_train.device)
					#exit()
					#centre = xp.ones_like(target_train)*centre		

					#print(prediction_train.shape, centre, target_train)
					#print(image_train.shape)
					#exit()
				
					# Calculate the loss with softmax_cross_entropy
					#print(prediction_train, centre, target_train)
					#loss = F.contrastive(prediction_train, centre, target_train)
					#print("loss")
					loss, losses = contrastive_loss(prediction_train, centre, target_train)
					loss_list_train.append(loss.item())
					
					bg_levels_list.extend(target_train_info[1])
					losses_list.extend(losses.tolist())
					
					accuracy = prediction_train_classified==target_train
					#print(target_train_info[0], target_train_info[1], prediction_train_classified, accuracy)
					#print("aaa", accuracy)
					train_accuracies.extend(accuracy.tolist())

					
					#train_loss_bg
					#print(len(optimizer.param_groups), optimizer.param_groups[0]["lr"], loss.item())
					
					#print(lr, loss)
					#lrs.append(lr)
					#avg_loss = beta * avg_loss + (1-beta)*loss.item()
					#smoothed_loss = avg_loss / (1 - beta**(i+1))
					#losses.append(smoothed_loss)
					#losses.append(loss.item())
					
					#print(prediction_train, prediction_train_classified, target_train, loss)
					#exit()				
					#print(loss.item(), len(train_iter))
					#print("loss", loss)
					#exit(0)		

				# Calculate the gradients in the network
				#model.cleargrads()
				#loss.backward()
				#print("backward")
				scaler.scale(loss).backward()

				# Update all the trainable parameters
				#optimizer.step()
				#print("step")
				scaler.step(optimizer)
				
				#print(scheduler.get_last_lr())
				#scheduler.step()
				#lr*=mult
				#optimizer.param_groups[0]["lr"] = lr
				
				#print("update")
				scaler.update()
				#scheduler.step()


				# Gather data and report
				running_loss += loss.item()
				if i == len(train_iter)-1:
				    last_loss = running_loss / len(train_iter) # loss per batch
				    print(f"  batch {i, trans_num} loss: {last_loss}")
				    #tb_x = epoch_index * len(train_iter) + i + 1
				    #writer.add_scalar('Loss/train', last_loss, tb_x)
				    running_loss = 0.

		#print(bg_levels_list, losses_list)

		for label, value in zip(bg_levels_list, losses_list):
			train_loss_bg[label].append(value)
		
		for label, value in zip(bg_levels_list, train_accuracies):
			train_acc_bg[label].append(value)
		
		for key in train_loss_bg:
			train_loss_bg[key] = np.mean(train_loss_bg[key])
			gtrain_loss_bg[key].SetPoint(gtrain_loss_bg[key].GetN(), gtrain_loss_bg[key].GetN(), train_loss_bg[key])
			train_acc_bg[key] = np.mean(train_acc_bg[key])
			gtrain_acc_bg[key].SetPoint(gtrain_acc_bg[key].GetN(), gtrain_acc_bg[key].GetN(), train_acc_bg[key])
			

		print("train losses per bg", train_loss_bg, "minibatches", len(loss_list_train))
		print("train acc", train_acc_bg)
		#c10 = arrays2canvas(lrs, losses)
		#wait4key()
		#exit()
		
		model.eval()

		# Check the validation accuracy of prediction after every epoch
		#if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch
			#with open("cur_centre.txt", "w") as cf:
			#	xp.savetxt(cf, model.centre.data[0])
			# Display the training loss
		print('epoch:{:02d} train_loss:{:.07f} '.format(
			epoch_index, float(last_loss)))
			
		gtl.SetPoint(gtl.GetN(), epoch_index, float(last_loss))

		bg_levels_list, losses_list = [], []

		with torch.no_grad():

			for i, data in enumerate(test_iter):
				#image_test, target_test, target_test = chainer.dataset.concat_examples(test_batch, device)
				#print("im", image_test.shape, target_test.shape)
				image_test_orig, target_test_info = data
				
				# Loop through affine transforms
				for trans_num in range(8):
					for imin in range(len(image_test_orig)):
						affine_transform(image_test_orig[imin], trans_num)		
						
					image_test, target_test = image_test_orig.to(device), target_test_info[0].to(device)
					
					with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
						# Forward the test data
						prediction_test, centre, prediction_test_classified = model.loss_predict(image_test, target_test)
						#print(prediction_test_classified, target_test)
						#wt = torch.where(prediction_test_classified!=target_test)[0]
						#w = wt.to("cpu")
						#print("a", target_test_info[1], "b", target_test_info[2])
						#print(np.dstack([np.array(target_test[wt].to("cpu")), np.array(target_test_info[2][0])[w], np.array(target_test_info[2][1])[w], np.array(target_test_info[2][2])[w]]))
						#print(prediction_test_classified, target_test)
						
						# Calculate the loss
						#loss_test = F.contrastive(prediction_test, centre, target_test)
						#print("loss tu", loss_test.array, centre, prediction_test, target_test)
						#exit()
						
						loss_test, losses = contrastive_loss(prediction_test, centre, target_test)
						loss_list_test.append(loss_test)
						
						bg_levels_list.extend(target_test_info[1])
						losses_list.extend(losses.tolist())
						# Calculate the accuracy
						accuracy = prediction_test_classified==target_test
						#print("in", prediction_test_classified, target_test, accuracy.astype(np.int))
						#print("aaa", accuracy)
						test_accuracies.extend(accuracy.tolist())
						#print(test_accuracies)
						#print(prediction_test_classified, target_test, prediction_test_classified==target_test, target_test_info[1])
													
						
						test_losses.append(loss_test.item())
						#print(i, trans_num, target_test_info[0], target_test_info[1], prediction_test_classified, accuracy)

						#exit()

		print("Test epoch", len(test_accuracies), len(losses_list))

		for label, value in zip(bg_levels_list, losses_list):
			test_loss_bg[label].append(value)

		for label, value in zip(bg_levels_list, test_accuracies):
			test_acc_bg[label].append(value)
		
		#print("test_acc_bg all", test_acc_bg, test_acc_bg[8], len(test_acc_bg[8]))
		
		for key in test_loss_bg:
			test_loss_bg[key] = np.mean(test_loss_bg[key])
			test_acc_bg[key] = np.mean(test_acc_bg[key])
			gtest_loss_bg[key].SetPoint(gtest_loss_bg[key].GetN(), gtest_loss_bg[key].GetN(), test_loss_bg[key])
			gtest_acc_bg[key].SetPoint(gtest_acc_bg[key].GetN(), gtest_acc_bg[key].GetN(), test_acc_bg[key])


		print("test losses per bg", test_loss_bg, "minibatches", len(loss_list_test))
		print("test accs per bg", test_acc_bg)
				

		#test_iter.reset()

		#print("out", test_accuracies)
		mean_accuracy = np.mean(test_accuracies)
		print(f'val_loss:{np.mean(test_losses):.7f} val_accuracy:{mean_accuracy}')

		out_dir = f"res_batchsize{args.batchsize}"

		try:
			os.mkdir(out_dir)
		except:
			pass


		gvl.SetPoint(gvl.GetN(), epoch_index, np.mean(test_losses))
		ga.SetPoint(ga.GetN(), epoch_index, mean_accuracy)
		cl.cd()
		gvl.Draw("AL*")
		gvl.SetMinimum(1e-9)
		gtl.Draw("same L")
		gtl.SetLineColor(2)
		gtl.SetMarkerColor(2)
		cl.SetLogy()
		pad_refresh()
		ca.cd()
		ga.Draw("AL*")
		ca.SetGridy()
		pad_refresh()
		
		"""
		print("aaa", [[el for el in train_loss_bg.values()]])
		mgltbg = arrays2multigraph([[el for el in train_loss_bg.values()]])
		mgltbg.Draw()
		pad_refresh()
		mglvbg = arrays2multigraph([[el for el in test_loss_bg.values()]])
		mglvbg.Draw()
		pad_refresh()		
		mgavbg = arrays2multigraph([[el for el in test_acc_bg.values()]])
		mgavbg.Draw()
		pad_refresh()
		"""

		bg_labels_test = sorted(test_loss_bg.keys(), key=lambda x: float(x))
		bg_labels_train = sorted(train_loss_bg.keys(), key=lambda x: float(x))

		cltbg.cd()
		for ibg,lab in enumerate(bg_labels_train):
			if ibg==0:
				gtrain_loss_bg[lab].Draw("AL*")
				gtrain_loss_bg[lab].SetMinimum(1e-9)
				gtrain_loss_bg[lab].SetMaximum(1)
				
			else:
				gtrain_loss_bg[lab].Draw("same L*")
			gtrain_loss_bg[lab].SetMarkerColor(root_colors[ibg])
			gtrain_loss_bg[lab].SetLineColor(root_colors[ibg])
			gtrain_loss_bg[lab].SetMarkerSize(4)
		cltbg.SetLogy()
		cltbg.SetGridy()
		pad_refresh()
		print(train_loss_bg)

		ctabg.cd()
		for ibg,lab in enumerate(bg_labels_train):
			if ibg==0:
				gtrain_acc_bg[lab].Draw("AL*")
				gtrain_acc_bg[lab].SetMinimum(0)
				gtrain_acc_bg[lab].SetMaximum(1)			
			else:
				gtrain_acc_bg[lab].Draw("same L*")
			gtrain_acc_bg[lab].SetMarkerColor(root_colors[ibg])
			gtrain_acc_bg[lab].SetLineColor(root_colors[ibg])
			gtrain_acc_bg[lab].SetMarkerSize(2)
		ctabg.SetGridy()
		pad_refresh()
		print(train_acc_bg)

		
		clvbg.cd()
		for ibg,lab in enumerate(bg_labels_test):
			if ibg==0:
				gtest_loss_bg[lab].Draw("AL*")
				gtest_loss_bg[lab].SetMinimum(1e-9)
				gtest_loss_bg[lab].SetMaximum(1)							
			else:
				gtest_loss_bg[lab].Draw("same L*")
			gtest_loss_bg[lab].SetMarkerColor(root_colors[ibg])
			gtest_loss_bg[lab].SetLineColor(root_colors[ibg])
			gtest_loss_bg[lab].SetMarkerSize(2)
		clvbg.SetLogy()
		clvbg.SetGridy()
		pad_refresh()
		print(test_loss_bg)
		
		cabg.cd()
		for ibg,lab in enumerate(bg_labels_test):
			print(ibg, lab, root_colors[ibg])
			if ibg==0:
				gtest_acc_bg[lab].Draw("AL*")
				gtest_acc_bg[lab].SetMinimum(0)
				gtest_acc_bg[lab].SetMaximum(1)			
			else:
				gtest_acc_bg[lab].Draw("same L*")
			gtest_acc_bg[lab].SetMarkerColor(root_colors[ibg])
			gtest_acc_bg[lab].SetLineColor(root_colors[ibg])
			gtest_acc_bg[lab].SetMarkerSize(2)
		cabg.SetGridy()
		pad_refresh()
		print(test_acc_bg)
		#wait4key()
		


		
		if (epoch_index%10==0 and epoch_index!=0) or epoch_index==0 or epoch_index==1 or epoch_index==2 or epoch_index==5 or epoch_index==6 or epoch_index==7 or epoch_index==8 or epoch_index==9 or epoch_index==11 or epoch_index==12 or epoch_index==13 or epoch_index==14 or epoch_index==15:
			print("saving")
			torch.save(model.state_dict(), out_dir+f'/curve_snapshot_epoch{epoch_index}.model')
			cl.SaveAs(out_dir+"/loss.png")
			cl.SaveAs(out_dir+"/loss.root")
			ca.SaveAs(out_dir+"/acc.png")
			ca.SaveAs(out_dir+"/acc.root")
			print("saved")
			#if epoch_index==1:
			#	exit()
				
	#			if mean_accuracy>max_accuracy:
	#				max_accuracy = mean_accuracy
	#				if max_accuracy>0.99:
	#					from chainer import serializers
	#					print("saving")
	#					serializers.save_npz(f'curve_snapshot_acc{max_accuracy}.model', mymlp)
	#					print("saved")


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        
    def forward(self, x0, x1, y, margin=None, reduce='mean'):
        #print(x0.shape, x1.shape, y.shape)
        #ctx.save_for_backward(x0, x1, y)
        if margin!=None:
	        self.margin = margin
        else:
                self.margin=gmargin
        self.reduce = reduce

        diff = x0 - x1
        #print("loss", x0, x1, diff, x0.shape, x1.shape)        
        dist_sq = torch.sum(diff ** 2, dim=1)
        dist = torch.sqrt(dist_sq)
        #print("in loss", dist, x0.shape, x1.shape, y.shape)
        d1 = dist
        mdist = self.margin - dist
        #print("loass margin", self.margin)
        dist = torch.max(mdist, torch.zeros_like(mdist))
        #print("dist2", dist)
        loss = (y * dist_sq + (1 - y) * dist * dist) * 0.5
        l1 = loss
        #exit()

        if reduce == 'mean':
            loss = torch.mean(loss)
            
        if torch.isnan(loss): 
            print(x0, x1, diff, dist_sq, d1, mdist, dist, l1, loss)
            exit()

        return loss, l1

def random_affine_transform(image):
	i = torch.randint(low=0, high=8, size=(1,))
	affine_transform(image, i[0])

def affine_transform(image, i):
	#print(i, image.shape)
	#c1 = array2canvas(np.copy(image[0, 60].numpy()))
	if i==0:
		pass
	elif i==1:
		image = torch.rot90(image, 1, dims=(2,3))
	elif i==2:
		image = torch.rot90(image, 2, dims=(2,3))
	elif i==3:
		image = torch.rot90(image, 3, dims=(2,3))
	elif i==4:
		# ToDo: check if it flips OK!
		image = torch.flip(image, (2,))
	elif i==5:
		image = torch.flip(image, (3,))
	elif i==6:
		image = torch.rot90(torch.flip(image, (2,)), 1, dims=(2,3))
	elif i==7:
		image = torch.rot90(torch.flip(image, (3,)), 1, dims=(2,3))

	#c2 = array2canvas(np.copy(image[0, 60].numpy()))
	#wait4key()

if __name__ == '__main__':

	main()




