#!/usr/bin/python
"""
How to randomise?
1. Add different poissonian backgrounds
2. Rotate the PDM
3. Change the starting frame
"""

# 65 bg in list

import uproot
import numpy as np
from etoshelpers import *

elves = [("/data/jem-euso-data/Mini-EUSO/191107/CPU_RUN_MAIN__2019_11_07__05_59_33__1100Cathode2FullPDMonlyself.root", 308, 383),
("/data/jem-euso-data/Mini-EUSO/191205/CPU_RUN_MAIN__2019_12_05__18_37_35__950Cathode3FullPDMonlyself.root", 180, 383), #2
("/data/jem-euso-data/Mini-EUSO/191205/CPU_RUN_MAIN__2019_12_05__18_41_09__950Cathode3FullPDMonlyself.root", 956, 1023),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 7347, 7392),
("/data/jem-euso-data/Mini-EUSO/200303/CPU_RUN_MAIN__2020_03_02__18_35_48__950Cathode3FullPDMonlyself.root", 1700, 1791),
#("/data/jem-euso-data/Mini-EUSO/203030/CPU_RUN_MAIN__2020_03_03__02_29_21__950Cathode3FullPDMonlyself.root", 2870, ), #elve
#("/data/jem-euso-data/Mini-EUSO/200303/CPU_RUN_MAIN__2020_03_03__02_29_21__950Cathode3FullPDMonlyself.root", 3772, ), #elve
#("/data/jem-euso-data/Mini-EUSO/200330/CPU_RUN_MAIN__2020_03_03__02_29_21__950Cathode3FullPDMonlyself.root", 4291, ), #elve
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 4100, 4277), #2
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_21__19_41_55__950Cathode3FullPDMonlyself.root", 1740, 1886), #2
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 300, 371),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 2500, 2687), #2
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_32_02__950Cathode3FullPDMonlyself.root", 1190, 1268),
#("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_32_02__950Cathode3FullPDMonlyself.root", 1860, 1919), #15 Doubtful and not on Zbyszek's list. Maybe remove?
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 820, 898),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_26__08_00_08__950Cathode3FullPDMonlyself.root", 2230, 2303),
("/data/jem-euso-data/Mini-EUSO/210108/CPU_RUN_MAIN__2021_01_09__07_43_31__950Cathode3FullPDMonlyself.root", 10020, 10111),
("/home/lewhoo/Mini-EUSO/220524/CPU_RUN_MAIN__2022_05_24__05_58_04__950Cathode3FullPDMonlyself.root", 128, 250)] #2 #19 
#("/home/lewhoo/Mini-EUSO/220524/CPU_RUN_MAIN__2022_05_24__05_58_04__950Cathode3FullPDMonlyself.root", 125, 250)]
# Detected with occnn
# CPU_RUN_MAIN__2021_01_09__07_45_42__950Cathode3FullPDMonlyself_l1_v10r2.root 4864 0.00083732407
# On the list: CPU_RUN_MAIN__2020_08_25__19_36_42__950Cathode3FullPDMonlyself_l1_v10r2.root 0 0.19341834
# CPU_RUN_MAIN__2020_07_22__01_39_58__950Cathode3FullPDMonlyself_l1_v11r1.root 1664 0.012728105
# CPU_RUN_MAIN__2020_08_26__00_18_17__950Cathode3FullPDMonlyself_l1_v11r1.root 6144 0.00083732407
# CPU_RUN_MAIN__2021_01_20__23_06_41__950Cathode3FullPDMonlyself_l1_v11r1.root 768 0.00083732407 # And previous packet, not detected
# Sun from ISS >-20.8
# Interesting, like super fast 2 frame eleve, but probably not an elve: CPU_RUN_MAIN__2021_01_05__03_00_17__950Cathode3FullPDMonlyself_l1_v11r1.root 1792 0.2500036
# CPU_RUN_MAIN__2020_08_21__07_15_20__950Cathode3FullPDMonlyself_l1_v11r1.root 640 0.46872866 - on the list, but high distance for very obvious elve. The previous packet has a very small distance. Probably should add into training set
# CPU_RUN_MAIN__2020_09_25__00_55_10__950Cathode3FullPDMonlyself_l1_v11r1.root 4224 0.5017768 also on the list with high distance


special_bg = [("/data/jem-euso-data/Mini-EUSO/191205/CPU_RUN_MAIN__2019_12_05__18_41_09__950Cathode3FullPDMonlyself.root", 512, 512+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 128, 128+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 384, 384+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 2688, 2688+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 7552, 7552+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 7680, 7680+128),
("/data/jem-euso-data/Mini-EUSO/200303/CPU_RUN_MAIN__2020_03_02__18_35_48__950Cathode3FullPDMonlyself.root", 384, 384+128), #
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 0, 0+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 1792, 1792+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 2432, 2432+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 10496, 10496+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 1024, 1024+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 1152, 1152+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 2048, 2048+128),#
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 4352, 4352+128),
#("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 4480, 4480+128),#/
#("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 4608, 4608+128),#/
#("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 5120, 5120+128),#/
#("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 5888, 5888+128),#/
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 6656, 6656+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 7424, 7424+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 7552, 7552+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 8704, 8704+128),
#("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 3072, 3072+128),#/
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 6400, 6400+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 7040, 7040+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 7808, 7808+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 7936, 7936+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 8064, 8064+128), ##
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 8704, 8704+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 8832, 8832+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_26__08_00_08__950Cathode3FullPDMonlyself.root", 3072, 3072+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_26__08_00_08__950Cathode3FullPDMonlyself.root", 4352, 4352+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_26__08_00_08__950Cathode3FullPDMonlyself.root", 4608, 4608+128),
("/home/lewhoo/Mini-EUSO/220524/CPU_RUN_MAIN__2022_05_24__05_58_04__950Cathode3FullPDMonlyself.root", 2304, 2304+128), #30
# New, 21.11.22
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 1536, 1536+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 2560, 2560+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 2816, 2816+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 3968, 3968+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 4480, 4480+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 5120, 5120+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 6272, 6272+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 6656, 6656+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 6784, 6784+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 7424, 7424+128),
("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 7808, 7808+128),
("/data/jem-euso-data/Mini-EUSO/200303/CPU_RUN_MAIN__2020_03_02__18_35_48__950Cathode3FullPDMonlyself.root", 640, 640+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 2560, 2560+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 3072, 3072+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 3200, 3200+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 3456, 3456+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 3584, 3584+128),
("/data/jem-euso-data/Mini-EUSO/200526/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself.root", 3968, 3968+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_21__19_41_55__950Cathode3FullPDMonlyself.root", 384, 384+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_21__19_41_55__950Cathode3FullPDMonlyself.root", 768, 768+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 1408, 1408+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 3072, 3072+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 3328, 3328+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 9088, 9088+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 9216, 9216+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 9728, 9728+128),
("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 10240, 10240+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_32_02__950Cathode3FullPDMonlyself.root", 2688, 2688+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 4608, 4608+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 5888, 5888+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 6400, 6400+128),
("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_26__08_00_08__950Cathode3FullPDMonlyself.root", 5632, 5632+128),
# More - can't make high validation with the ones below for some reason
#("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 1920, 1920+128),
##("/data/jem-euso-data/Mini-EUSO/191230/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself.root", 5504, 5504+128),
#("/data/jem-euso-data/Mini-EUSO/200721/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself.root", 11136, 11136+128),
##("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_32_02__950Cathode3FullPDMonlyself.root", 3328, 3328+128),
#("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself.root", 2560, 2560+128),
#("/data/jem-euso-data/Mini-EUSO/200825/CPU_RUN_MAIN__2020_08_26__08_00_08__950Cathode3FullPDMonlyself.root", 5888, 5888+128),
# Bg from whole session
("/home/lewhoo/Mini-EUSO/S1_elves/other_bg/1/CPU_RUN_MAIN__2019_12_30__18_48_27__950Cathode3FullPDMonlyself_l1_v11r2.root", 2816, 2816+128),
("/home/lewhoo/Mini-EUSO/S1_elves/other_bg/1/CPU_RUN_MAIN__2019_12_31__05_21_08__950Cathode3FullPDMonlyself_l1_v11r2.root", 1920, 1920+128),
("/home/lewhoo/Mini-EUSO/S1_elves/other_bg/1/CPU_RUN_MAIN__2019_12_30__21_08_03__950Cathode3FullPDMonlyself_l1_v11r2.root", 10240, 10240+128),
("/home/lewhoo/Mini-EUSO/S1_elves/other_bg/1/CPU_RUN_MAIN__2019_12_31__03_19_54__950Cathode3FullPDMonlyself_l1_v11r2.root", 8192, 8192+128)
]

# Change the list of files to the flattened one
elves1 = []
for el in elves:
	fn = el[0]
	if "PU_RUN_MAIN__2022_05_24__05_58_04__" not in el[0] and "_l1_" not in el[0]: elves1.append((("/home/lewhoo/Mini-EUSO/S1_elves/1/"+el[0].split("/")[-1]).split(".root")[0]+"_l1_v11r2.root", el[1], el[2]))
	else: elves1.append(el)

elves = elves1

elves1 = []
for el in special_bg:
	fn = el[0]
	if "PU_RUN_MAIN__2022_05_24__05_58_04__" not in el[0] and "_l1_" not in el[0]: elves1.append((("/home/lewhoo/Mini-EUSO/S1_elves/1/"+el[0].split("/")[-1]).split(".root")[0]+"_l1_v11r2.root", el[1], el[2]))
	else: elves1.append(el)

special_bg = elves1


#import random
#random.shuffle(special_bg)


normalise = 0

valid_point = 11
#valid_point = 30
valid_point_special_bg = 19
#"""
elves_data = []
bg_data = []
elves_data_d, bg_data_d = [], []

bg_cnt=0
elf_cnt=0

# Read most of the elves - the rest remains for unbiased validation
for elf in elves[:valid_point]:
	with uproot.open(f"{elf[0]}:tevent") as t:
		ev_len = elf[2]-elf[1]
		first_frame = elf[1]//128*128
		print(ev_len//128, ev_len)
		pc = t["photon_count_data"].array().to_numpy()
		
		# Normalise the one file that is not flattened
		if "PU_RUN_MAIN__2022_05_24__05_58_04__" in elf[0]:
			flat = np.load("/home/lewhoo/Mini-EUSO/220524/pdm.npy")
			flat = np.swapaxes(flat, 1, 0)
			flat[flat<0.05]=1
			pc/=flat
		
		# Treat each packet of the elf as a separate event
		for pkt in range(ev_len//128+1):
			print(elf)
			print("pkt", pkt)
			print("cut", first_frame+pkt*128,first_frame+(pkt+1)*128)
			elf_data = pc[first_frame+pkt*128:first_frame+(pkt+1)*128,0,0]
			np_elf_data = elf_data.astype(np.float32)
			# Normalise
			if normalise==1: np_elf_data = (np_elf_data-np.mean(np_elf_data))/np.std(np_elf_data)
			elif normalise==2: np_elf_data /= np.max(np_elf_data[np_elf_data!=255])
			elves_data.append(np.copy(np_elf_data.astype(np.float32)))
			elves_data_d.append((elf[0], first_frame+pkt*128, first_frame+(pkt+1)*128))
			
#			x = elves_data[-1]
#			for j in range(55,128):
#				h = array2histogram(x[j])
#				h.SetMinimum(0)
#				h.SetMaximum(30)
#				h.Draw("colz")
#				print(j, t)
#				pad_refresh()
#				br = wait4key()
#				if br: break
			
			
			last_frame = first_frame+(pkt+1)*128
			elf_cnt+=1

			
		# Assume that the packet before and after the elve is clean
		print(elf, first_frame, last_frame)
		# Just one bg, to have the same amount of bg as elves
		#bg_data.append(t["photon_count_data"].array()[first_frame-128:first_frame,0,0].to_numpy().astype(np.float32))
		#np_bg_data = t["photon_count_data"].array()[first_frame-128:first_frame,0,0].to_numpy()
		#np_bg_data = (np_bg_data-np.mean(np_bg_data))/np.std(np_bg_data)
		#bg_data.append(np.copy(np_bg_data))
		if first_frame>=128: np_bg_data = pc[first_frame-128:first_frame,0,0].astype(np.float32)
		else: np_bg_data = pc[last_frame+128:last_frame+128*2,0,0].astype(np.float32)
		if normalise==1: np_bg_data = (np_bg_data-np.mean(np_bg_data))/np.std(np_bg_data)
		elif normalise==2: np_bg_data /= np.max(np_bg_data[np_bg_data!=255])
		bg_data.append(np.copy(np_bg_data))
		bg_data_d.append((elf[0], last_frame+128,last_frame+128*2))
		bg_cnt+=1

print(len(bg_data))

print("*************SPECIAL BG", len(special_bg))

# Add special bg
#for elf in special_bg[:valid_point_special_bg]:
for elf in special_bg:
	with uproot.open(f"{elf[0]}:tevent") as t:
		pc = t["photon_count_data"].array().to_numpy()
		
		# Normalise the one file that is not flattened
		if "PU_RUN_MAIN__2022_05_24__05_58_04__" in elf[0]:
			flat = np.load("/home/lewhoo/Mini-EUSO/220524/pdm.npy")
			flat = np.swapaxes(flat, 1, 0)
			flat[flat<0.05]=1
			pc/=flat		
		
		elf_data = pc[elf[1]:elf[2],0,0]
		np_elf_data = elf_data.astype(np.float32)
		# Normalise
		if normalise==1: np_elf_data = (np_elf_data-np.mean(np_elf_data))/np.std(np_elf_data)
		elif normalise==2: np_elf_data /= np.max(np_elf_data[np_elf_data!=255])
		bg_data.append(np.copy(np_elf_data.astype(np.float32)))
		bg_data_d.append((elf[0], elf[1], elf[2]))
		bg_cnt+=1

		
#import ROOT
#c = ROOT.TCanvas()
			
# Generate elves samples
# add random poisson bg, random orientation, flipping (later roll the time window to the previous/next packet, median subtract an elf and add a clean bg normalised)
elves_samples = []
for j,elf in enumerate(elves_data):
	print(type(elf))
	# Assuming bg does not change in the packet... which is not necessarily true, but should not be that bad in D1
	#med = np.median(elf)
	# First the original elf
	elves_samples.append((np.copy(elf),1, 0, elves_data_d[j]))	
	#print(elf, elf.dtype)	
	# Loop through random backgrounds
	for i in range(10):
		# Generate slightly randomised samples
		bg_lev = 0.2
		bg = np.random.randn(*([8]+list(elf.shape))).astype(np.float32)*bg_lev+1
#		cur_elf = np.rint((elf.astype(np.float32)+1)*bg-1)
		cur_elf = (elf.astype(np.float32)+1)*bg-1
		cur_elf[cur_elf<0]=0

		elves_samples.append((np.copy(cur_elf[0]),1, bg_lev, elves_data_d[j]))
		# Rotations
		elves_samples.append((np.copy(np.rot90(cur_elf[1], 1, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		elves_samples.append((np.copy(np.rot90(cur_elf[2], 2, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		elves_samples.append((np.copy(np.rot90(cur_elf[3], 3, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		# Flips
		elves_samples.append((np.copy(np.flip(cur_elf[4], 1)),1, bg_lev, elves_data_d[j]))
		elves_samples.append((np.copy(np.flip(cur_elf[5], 2)),1, bg_lev, elves_data_d[j]))
		# Diag flips
		elves_samples.append((np.copy(np.rot90(np.flip(cur_elf[6], 1), 1, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		elves_samples.append((np.copy(np.rot90(np.flip(cur_elf[7], 1), 3, axes=(1,2))),1, bg_lev, elves_data_d[j]))

		
# Do the same for background
for j, elf in enumerate(bg_data):
	# Assuming bg does not change in the packet... which is not necessarily true, but should not be that bad in D1
	med = np.median(elf)
	#print("tu", type(elf), elf.shape, med)
	# First the original bg
	elves_samples.append((np.copy(elf),0, 0, bg_data_d[j]))

	# Loop through random backgrounds
	for i in range(3):
		# Generate slightly randomised samples
		bg_lev=0.2
		bg = np.random.randn(*([8]+list(elf.shape))).astype(np.float32)*bg_lev+1
#		cur_elf = np.rint((elf.astype(np.float32)+1)*bg-1)
		cur_elf = (elf.astype(np.float32)+1)*bg-1
		cur_elf[cur_elf<0]=0	
		
		elves_samples.append((np.copy(cur_elf[0]),0, bg_lev, bg_data_d[j]))
		# Rotations
		elves_samples.append((np.copy(np.rot90(cur_elf[1], 1, axes=(1,2))),0, bg_lev, bg_data_d[j]))		
		elves_samples.append((np.copy(np.rot90(cur_elf[2], 2, axes=(1,2))),0, bg_lev, bg_data_d[j]))	
		elves_samples.append((np.copy(np.rot90(cur_elf[3], 3, axes=(1,2))),0, bg_lev, bg_data_d[j]))
		# Flips		
		elves_samples.append((np.copy(np.flip(cur_elf[4], 1)),0, bg_lev, bg_data_d[j]))
		continue		
		elves_samples.append((np.copy(np.flip(cur_elf[5], 2)),0, bg_lev, bg_data_d[j]))
		continue		
		# Diag flips
		elves_samples.append((np.copy(np.rot90(np.flip(cur_elf[6], 1), 1, axes=(1,2))),0, bg_lev, bg_data_d[j]))
		elves_samples.append((np.copy(np.rot90(np.flip(cur_elf[7], 1), 3, axes=(1,2))),0, bg_lev, bg_data_d[j]))

# Count the labels
labels = np.array([el[1] for el in elves_samples])
print("labels", np.count_nonzero(labels==0), np.count_nonzero(labels==1))
print(bg_cnt, elf_cnt)
#exit()

# Add dimension corresponding to channels - needed for CNN
elves_samples = [(np.expand_dims(el[0], 0).astype(np.float32), el[1:]) for el in elves_samples]

# Shuffle the array to mix bg and elves
import random
random.shuffle(elves_samples)

import pickle
with open("elves_samples.pk", "wb") as f:
	pickle.dump(elves_samples, f)

#"""	

# ************** Now repeat everything for the remaining elves - unbiased validation file

elves_data = []
bg_data = []
elves_data_d, bg_data_d = [], []

#import ROOT
#from etoshelpers import *
#c1 = ROOT.TCanvas()

# Read most of the elves - the rest remains for unbiased validation
for elf in elves[valid_point:]:
	print(elf)
	with uproot.open(f"{elf[0]}:tevent") as t:
		ev_len = elf[2]-elf[1]
		first_frame = elf[1]//128*128
		print(ev_len//128, ev_len)
		pc = t["photon_count_data"].array().to_numpy()
		
		# Normalise the one file that is not flattened
		if "PU_RUN_MAIN__2022_05_24__05_58_04__" in elf[0]:
			flat = np.load("/home/lewhoo/Mini-EUSO/220524/pdm.npy")
			flat = np.swapaxes(flat, 1, 0)
			flat[flat<0.05]=1
			pc/=flat		
		
		# Treat each packet of the elf as a separate event
		for pkt in range(ev_len//128+1):		
			elf_data = pc[first_frame+pkt*128:first_frame+(pkt+1)*128,0,0]	
			np_elf_data = elf_data
			if elf_data.shape[0]==0:
				print(elf_data.shape, "elf")
				exit()
			# Normalise
			if normalise==1: np_elf_data = (np_elf_data-np.mean(np_elf_data))/np.std(np_elf_data)
			elif normalise==2: np_elf_data /= np.max(np_elf_data[np_elf_data!=255])
			elves_data.append(np.copy(np_elf_data.astype(np.float32)))
			elves_data_d.append((elf[0], first_frame+pkt*128, first_frame+(pkt+1)*128))	
			#print(elf_data)
			last_frame = first_frame+(pkt+1)*128
					
		# Assume that the packet before and after the elve is clean
		#print(first_frame, last_frame)
		if first_frame>=128: np_bg_data = pc[first_frame-128:first_frame,0,0].astype(np.float32)
		else: np_bg_data = pc[last_frame+128*2:last_frame+128*3,0,0].astype(np.float32)
		if normalise==1: np_bg_data = (np_bg_data-np.mean(np_bg_data))/np.std(np_bg_data)
		elif normalise==2: np_bg_data /= np.max(np_bg_data[np_bg_data!=255])
		bg_data.append(np.copy(np_bg_data))
		bg_data_d.append((elf[0], last_frame+128*2, last_frame+128*3))
		if np_bg_data.shape[0]==0:
			print(np_bg_data.shape, "bg1")
			exit()
		
		if last_frame+128*2<=pc.shape[1]: np_bg_data = pc[last_frame+128:last_frame+128*2,0,0].astype(np.float32)
		else: np_bg_data = pc[first_frame-128*2:first_frame-128,0,0].astype(np.float32)
		if normalise==1: np_bg_data = (np_bg_data-np.mean(np_bg_data))/np.std(np_bg_data)
		elif normalise==2: np_bg_data /= np.max(np_bg_data[np_bg_data!=255])
		if np_bg_data.shape[0]!=0: 
			bg_data.append(np.copy(np_bg_data))
			bg_data_d.append((elf[0], first_frame-128*2, first_frame-128))

print("*************SPECIAL BG", len(special_bg[valid_point_special_bg:]), valid_point_special_bg)

# Add special bg
for elf in special_bg[valid_point_special_bg:]:
	with uproot.open(f"{elf[0]}:tevent") as t:
		pc = t["photon_count_data"].array().to_numpy()
		
		# Normalise the one file that is not flattened
		if "PU_RUN_MAIN__2022_05_24__05_58_04__" in elf[0]:
			flat = np.load("/home/lewhoo/Mini-EUSO/220524/pdm.npy")
			flat = np.swapaxes(flat, 1, 0)
			flat[flat<0.05]=1
			pc/=flat		
		
		elf_data = pc[elf[1]:elf[2],0,0]
		np_elf_data = elf_data.astype(np.float32)
		# Normalise
		if normalise==1: np_elf_data = (np_elf_data-np.mean(np_elf_data))/np.std(np_elf_data)
		elif normalise==2: np_elf_data /= np.max(np_elf_data[np_elf_data!=255])
		bg_data.append(np.copy(np_elf_data.astype(np.float32)))
		bg_data_d.append((elf[0], elf[1], elf[2]))
		
			
# Generate elves samples
# add random poisson bg, random orientation, flipping (later roll the time window to the previous/next packet, median subtract an elf and add a clean bg normalised)
elves_samples = []
for j,elf in enumerate(elves_data):
	print(type(elf))
	# Assuming bg does not change in the packet... which is not necessarily true, but should not be that bad in D1
	med = np.median(elf)
	# First the original elf
	elves_samples.append((np.copy(elf),1, 0, elves_data_d[j]))	
	# Loop through random backgrounds
	for i in range(10):
		# Generate slightly randomised samples
		bg_lev=0.2
		bg = np.random.randn(*([8]+list(elf.shape))).astype(np.float32)*bg_lev+1
#		cur_elf = np.rint((elf.astype(np.float32)+1)*bg-1)
		cur_elf = (elf.astype(np.float32)+1)*bg-1
		cur_elf[cur_elf<0]=0
		
		elves_samples.append((np.copy(cur_elf[0]),1, bg_lev, elves_data_d[j]))
		
#		from etoshelpers import *
##		c = create_fill_canvas_with_histogram_1D(np.ravel(elves_samples[-1][0]))
#		c = array2canvas(elves_samples[-1][0][70])
#		pad_refresh()
##		c1 = create_fill_canvas_with_histogram_1D(np.ravel(elf))
#		c1 = array2canvas(elf[70])
#		pad_refresh()
#		print(np.count_nonzero(elves_samples[-1][0]==0), np.count_nonzero(elf==0))
#		wait4key()
		
		# Rotations
		elves_samples.append((np.copy(np.rot90(cur_elf[1], 1, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		elves_samples.append((np.copy(np.rot90(cur_elf[2], 2, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		elves_samples.append((np.copy(np.rot90(cur_elf[3], 3, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		# Flips
		elves_samples.append((np.copy(np.flip(cur_elf[4], 1)),1, bg_lev, elves_data_d[j]))
		elves_samples.append((np.copy(np.flip(cur_elf[5], 2)),1, bg_lev, elves_data_d[j]))
		# Diag flips
		elves_samples.append((np.copy(np.rot90(np.flip(cur_elf[6], 1), 1, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		elves_samples.append((np.copy(np.rot90(np.flip(cur_elf[7], 1), 3, axes=(1,2))),1, bg_lev, elves_data_d[j]))
		
# Do the same for background
for j, elf in enumerate(bg_data):
	# Assuming bg does not change in the packet... which is not necessarily true, but should not be that bad in D1
	#med = np.median(elf)
	#print("tu", type(elf), elf.shape, med)
	# First the original bg
	elves_samples.append((np.copy(elf),0, 0, bg_data_d[j]))
	# Loop through random backgrounds
	for i in range(3):
		# Generate slightly randomised samples
		bg_lev=0.2
		bg = np.random.randn(*([8]+list(elf.shape))).astype(np.float32)*bg_lev+1
#		cur_elf = np.rint((elf.astype(np.float32)+1)*bg-1)
		cur_elf = (elf.astype(np.float32)+1)*bg-1
		cur_elf[cur_elf<0]=0
	
		# From now on, with randomised values		
		elves_samples.append((np.copy(cur_elf[0]),0, bg_lev, bg_data_d[j]))
		# Rotations
		elves_samples.append((np.copy(np.rot90(cur_elf[1], 1, axes=(1,2))),0, bg_lev, bg_data_d[j]))
		elves_samples.append((np.copy(np.rot90(cur_elf[2], 2, axes=(1,2))),0, bg_lev, bg_data_d[j]))
		elves_samples.append((np.copy(np.rot90(cur_elf[3], 3, axes=(1,2))),0, bg_lev, bg_data_d[j]))
		continue		
		# Flips
		elves_samples.append((np.copy(np.flip(cur_elf[4], 1)),0, bg_lev, bg_data_d[j]))
		elves_samples.append((np.copy(np.flip(cur_elf[5], 2)),0, bg_lev, bg_data_d[j]))
		# Diag flips
		elves_samples.append((np.copy(np.rot90(np.flip(cur_elf[6], 1), 1, axes=(1,2))),0, bg_lev, bg_data_d[j]))
		elves_samples.append((np.copy(np.rot90(np.flip(cur_elf[7], 1), 3, axes=(1,2))),0, bg_lev, bg_data_d[j]))

# Add dimension corresponding to channels - needed for CNN
elves_samples = [(np.expand_dims(el[0], 0).astype(np.float32), el[1:]) for el in elves_samples]

# Shuffle the array to mix bg and elves
import random
random.shuffle(elves_samples)

import pickle
with open("elves_samples_validation.pk", "wb") as f:
	pickle.dump(elves_samples, f)

#compress_pickle("elves_samples_validation.pk", elves_samples)

