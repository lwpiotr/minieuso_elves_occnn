#!/usr/bin/python

import glob
import subprocess
import ROOT

nf = open("tle_occnn_results_sunfiltered.txt", "w")

#flist = glob.glob("*/tle_results_occnn.txt")

#for i,fl in enumerate(flist):
fl = "tle_occnn_results.txt"
with open(fl, "r") as f:
	lines = f.readlines()
	cand_cnt = len([l for l in lines if float(l.split()[2])<0.2])
	print(fl, len(lines))
	el_num=0
	for j,l in enumerate(lines):
		filename, frame_num, dist = l.split()
		frame_num = int(frame_num)
		dist = float(dist)
		rf = ROOT.TFile(filename)
		t = rf.Get("tevent")
		t.GetEntry(frame_num)
		# Basically check if the root file has the Sun to ISS angle
		try:
			if t.Sun_from_ISS_alt<-20.8:
				print(l.strip(), file=nf)
		except:
			print(l.strip(), file=nf)

