#!/usr/bin/python
"""
How to randomise?
1. Add different poissonian backgrounds
2. Rotate the PDM
3. Change the starting frame
"""

import uproot
import numpy as np
from etoshelpers import *

add_standard_bg = True

# 39 elve samples
elves = [("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2019_11_07__05_59_33__1100Cathode2FullPDMonlyself_l1_v11r2.root", 256),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2019_12_05__18_37_35__950Cathode3FullPDMonlyself_l1_v11r2.root", 128),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2019_12_05__18_37_35__950Cathode3FullPDMonlyself_l1_v11r2.root", 256),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2019_12_05__18_41_09__950Cathode3FullPDMonlyself_l1_v11r2.root", 896), 
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2019_12_30__18_15_11__950Cathode3FullPDMonlyself_l1_v11r2.root", 7296),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_03_02__18_35_48__950Cathode3FullPDMonlyself_l1_v11r2.root", 1664),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_03_03__02_29_21__950Cathode3FullPDMonlyself_l1_v11r2.root", 2816),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_03_03__02_29_21__950Cathode3FullPDMonlyself_l1_v11r2.root", 2944),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_03_03__02_29_21__950Cathode3FullPDMonlyself_l1_v11r2.root", 3712),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_03_03__02_29_21__950Cathode3FullPDMonlyself_l1_v11r2.root", 4224),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself_l1_v11r2.root", 4096),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_05_26__07_38_55__950Cathode3FullPDMonlyself_l1_v11r2.root", 4224),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_06_29__19_28_57__950Cathode3FullPDMonlyself_l1_v11r1.root", 896),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_07_21__19_41_55__950Cathode3FullPDMonlyself_l1_v11r2.root", 1664),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_07_21__19_41_55__950Cathode3FullPDMonlyself_l1_v11r2.root", 1792),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_07_22__01_39_58__950Cathode3FullPDMonlyself_l1_v11r2.root", 1664), # In principle continues to the next packet, but rings almost invisible
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself_l1_v11r2.root", 256),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself_l1_v11r2.root", 2432),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_07_22__07_59_55__950Cathode3FullPDMonlyself_l1_v11r2.root", 2560),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_07_27__21_30_53__950Cathode3FullPDMonlyself_l1_v11r2.root", 896),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_07_27__21_30_53__950Cathode3FullPDMonlyself_l1_v11r2.root", 896+128),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_08_21__07_15_20__950Cathode3FullPDMonlyself_l1_v11r2.root", 512),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_08_21__07_15_20__950Cathode3FullPDMonlyself_l1_v11r2.root", 512+128),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_08_25__19_32_02__950Cathode3FullPDMonlyself_l1_v11r2.root", 1152),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_08_25__19_34_27__950Cathode3FullPDMonlyself_l1_v11r2.root", 768), # Only two frames really visible on the next packet, so ignoring
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_08_25__19_36_42__950Cathode3FullPDMonlyself_l1_v11r2.root", 0),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_08_26__00_18_17__950Cathode3FullPDMonlyself_l1_v11r2.root", 6144),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_08_26__08_00_08__950Cathode3FullPDMonlyself_l1_v11r2.root", 2176),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_09_15__00_25_23__950Cathode3FullPDMonlyself_l1_v11r2.root", 8320), # In principle continues to the next packet, but rings almost invisible
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_09_24__18_43_42__950Cathode3FullPDMonlyself_l1_v11r2.root", 5760),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_09_24__18_43_42__950Cathode3FullPDMonlyself_l1_v11r2.root", 5760+128),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_09_25__00_55_10__950Cathode3FullPDMonlyself_l1_v11r2.root", 4096),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_09_25__00_55_10__950Cathode3FullPDMonlyself_l1_v11r2.root", 4096+128),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2020_12_10__00_49_41__950Cathode3FullPDMonlyself_l1_v11r2.root", 2560),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2021_01_09__06_07_24__950Cathode3FullPDMonlyself_l1_v11r2.root", 1408),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2021_01_09__07_43_31__950Cathode3FullPDMonlyself_l1_v11r2.root", 9984),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2021_01_09__07_45_42__950Cathode3FullPDMonlyself_l1_v11r2.root", 4864),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2021_01_20__23_06_41__950Cathode3FullPDMonlyself_l1_v11r2.root", 640),
("/home/lewhoo/data/Mini-EUSO/found_elves_files/CPU_RUN_MAIN__2021_01_20__23_06_41__950Cathode3FullPDMonlyself_l1_v11r2.root", 768)] 

# 66 bg in list
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
("/home/lewhoo/Mini-EUSO/S1_elves/other_bg/1/CPU_RUN_MAIN__2019_12_31__03_19_54__950Cathode3FullPDMonlyself_l1_v11r2.root", 8192, 8192+128) # Removing, because 39 elves * 8 (affine trasnforms) * 5 bg levels = 1560, and that's divisible by 65
]


elves1 = []
for el in special_bg:
	fn = el[0]
	if "PU_RUN_MAIN__2022_05_24__05_58_04__" not in el[0] and "_l1_" not in el[0]: elves1.append((("/home/lewhoo/Mini-EUSO/S1_elves/1/"+el[0].split("/")[-1]).split(".root")[0]+"_l1_v11r2.root", el[1], el[2]))
	else: elves1.append(el)

special_bg = elves1


elve_start_frames = [(el[0], el[1]) for el in elves]

#import random
#random.shuffle(special_bg)


normalise = 0

#"""
elves_data = []
bg_data = []
elves_data_d, bg_data_d = [], []

bg_cnt=0
elf_cnt=0

additional_bg_frames = []

# Read most of the elves - the rest remains for unbiased validation
for elf in elves:
	with uproot.open(f"{elf[0]}:tevent") as t:
		pc = t["photon_count_data"].array()
				
		elf_data = pc[elf[1]:elf[1]+128,0,0].to_numpy()

		# Normalise the one file that is not flattened
		if "PU_RUN_MAIN__2022_05_24__05_58_04__" in elf[0]:
			flat = np.load("/home/lewhoo/Mini-EUSO/220524/pdm.npy")
			flat = np.swapaxes(flat, 1, 0)
			flat[flat<0.05]=1
			elf_data/=flat
		
		np_elf_data = elf_data.astype(np.float32)
		# Normalise
		elves_data.append(np.copy(np_elf_data.astype(np.float32)))
		elves_data_d.append((elf[0], elf[1], elf[1]+128))
		
		# The bg from "2021_01_20__23_06_41" somehow cause the CNN to go to NaNs...
		if add_standard_bg and "2021_01_20__23_06_41" not in elf[0]:
			if elf[1]+4*128<=t.num_entries and (el[0], elf[1]+3*128) not in elve_start_frames:
				special_bg.append((elf[0], elf[1]+3*128, elf[1]+4*128))
				additional_bg_frames.append((elf[0], elf[1]+3*128))
				print(t.num_entries, elf[0], elf[1]+3*128, elf[1]+4*128)
			#if elf[1]-3*128>=0 and (el[0], elf[1]-3*128) not in elve_start_frames and (elf[0], elf[1]-3*128) not in additional_bg_frames:
			#	special_bg.append((elf[0], elf[1]-3*128, elf[1]-2*128))
			#	print(t.num_entries, elf[0], elf[1]-3*128, elf[1]-2*128)
			
		elf_cnt+=1
#exit()
print("*************SPECIAL BG", len(special_bg))

print(special_bg[100:106], len(special_bg))
#exit()

# Add special bg
#for elf in special_bg[:valid_point_special_bg]:
#100, 106
for elf in special_bg[:100]:
	with uproot.open(f"{elf[0]}:tevent") as t:
		#pc = t["photon_count_data"].array().to_numpy()
		pc = t["photon_count_data"].array()
				
		elf_data = pc[elf[1]:elf[2],0,0].to_numpy()
		
		# Normalise the one file that is not flattened
		if "PU_RUN_MAIN__2022_05_24__05_58_04__" in elf[0]:
			flat = np.load("/home/lewhoo/Mini-EUSO/220524/pdm.npy")
			flat = np.swapaxes(flat, 1, 0)
			flat[flat<0.05]=1
			elf_data/=flat				
		
		np_elf_data = elf_data.astype(np.float32)
		bg_data.append(np.copy(np_elf_data.astype(np.float32)))
		bg_data_d.append((elf[0], elf[1], elf[2]))
		bg_cnt+=1

print(elf_cnt, bg_cnt)
		
#import ROOT
#c = ROOT.TCanvas()
			
# Generate elves samples
# add random poisson bg, random orientation, flipping (later roll the time window to the previous/next packet, median subtract an elf and add a clean bg normalised)
elves_samples = []
for j,elf in enumerate(elves_data):
	elves_samples.append((np.copy(elf),1, "0", elves_data_d[j]))
	# Append also 4 levels of bg - to get to 195 elves and bg
#	bg = np.random.randn(*([4]+list(elf.shape))).astype(np.float32)
#	elves_samples.append((np.copy(elf)*(bg[0]*0.2+1),1, "0.2", elves_data_d[j]))
#	elves_samples.append((np.copy(elf)*(bg[1]*0.4+1),1, "0.4", elves_data_d[j]))
#	elves_samples.append((np.copy(elf)*(bg[2]*0.8+1),1, "0.8", elves_data_d[j]))
#	elves_samples.append((np.copy(elf)*(bg[3]*1.4+1),1, "1.4", elves_data_d[j]))
#	elves_samples.append((np.copy(elf)*(bg[0]*0.4+1),1, "0.4", elves_data_d[j]))
#	elves_samples.append((np.copy(elf)*(bg[1]*0.8+1),1, "0.8", elves_data_d[j]))
#	elves_samples.append((np.copy(elf)*(bg[2]*1.6+1),1, "1.6", elves_data_d[j]))
#	elves_samples.append((np.copy(elf)*(bg[3]*3.2+1),1, "3.2", elves_data_d[j]))

	bg_mult=0.05
#	bg = np.random.poisson(1*bg_mult, size=elf.shape)
#	elves_samples.append(((np.copy(elf)+bg)/(1*bg_mult),1, str(1*bg_mult), elves_data_d[j]))
#	bg = np.random.poisson(2*bg_mult, size=elf.shape)
#	elves_samples.append(((np.copy(elf)+bg)/(2*bg_mult),1, str(2*bg_mult), elves_data_d[j]))
#	bg = np.random.poisson(3*bg_mult, size=elf.shape)
#	elves_samples.append(((np.copy(elf)+bg)/(3*bg_mult),1, str(3*bg_mult), elves_data_d[j]))
#	bg = np.random.poisson(4*bg_mult, size=elf.shape)
#	elves_samples.append(((np.copy(elf)+bg)/(4*bg_mult),1, str(4*bg_mult), elves_data_d[j]))
	bg = np.random.poisson(1*bg_mult, size=elf.shape)
	elves_samples.append(((np.copy(elf)+bg),1, f"{1*bg_mult:.2f}", elves_data_d[j]))
	bg = np.random.poisson(2*bg_mult, size=elf.shape)
	elves_samples.append(((np.copy(elf)+bg),1, f"{2*bg_mult:.2f}", elves_data_d[j]))
	bg = np.random.poisson(3*bg_mult, size=elf.shape)
	elves_samples.append(((np.copy(elf)+bg),1, f"{3*bg_mult:.2f}", elves_data_d[j]))
	bg = np.random.poisson(4*bg_mult, size=elf.shape)
	elves_samples.append(((np.copy(elf)+bg),1, f"{4*bg_mult:.2f}", elves_data_d[j]))


	elves_samples[-1][0][elves_samples[-1][0]<0]=0
	elves_samples[-2][0][elves_samples[-2][0]<0]=0
	elves_samples[-3][0][elves_samples[-3][0]<0]=0
	elves_samples[-4][0][elves_samples[-4][0]<0]=0


bg_samples = []		
# Do the same for background
for j, elf in enumerate(bg_data):
	bg_samples.append((np.copy(elf),0, "0", bg_data_d[j]))
	# Append also 2 levels of bg - to get to 195 elves and bg
	bg = np.random.randn(*([4]+list(elf.shape))).astype(np.float32)
#	bg_samples.append((np.copy(elf)*(bg[0]*0.2+1),0, "0.2", bg_data_d[j]))
#	bg_samples.append((np.copy(elf)*(bg[1]*0.4+1),0, "0.4", bg_data_d[j]))
#	bg_samples.append((np.copy(elf)*(bg[2]*0.8+1),0, "0.8", bg_data_d[j]))
#	bg_samples.append((np.copy(elf)*(bg[3]*1.4+1),0, "1.4", bg_data_d[j]))
#	bg_samples.append((np.copy(elf)*(bg[0]*0.4+1),0, "0.4", bg_data_d[j]))
#	bg_samples.append((np.copy(elf)*(bg[1]*0.8+1),0, "0.8", bg_data_d[j]))
#	bg_samples.append((np.copy(elf)*(bg[2]*1.6+1),0, "1.6", bg_data_d[j]))
#	bg_samples.append((np.copy(elf)*(bg[3]*3.2+1),0, "3.2", bg_data_d[j]))

	bg_mult=0.05
#	bg = np.random.poisson(1*bg_mult, size=elf.shape)
#	bg_samples.append(((np.copy(elf)+bg)/(1*bg_mult),0, str(1*bg_mult), bg_data_d[j]))
#	bg = np.random.poisson(2*bg_mult, size=elf.shape)
#	bg_samples.append(((np.copy(elf)+bg)/(2*bg_mult),0, str(2*bg_mult), bg_data_d[j]))
#	bg = np.random.poisson(3*bg_mult, size=elf.shape)
#	bg_samples.append(((np.copy(elf)+bg)/(3*bg_mult),0, str(3*bg_mult), bg_data_d[j]))
#	bg = np.random.poisson(4*bg_mult, size=elf.shape)
#	bg_samples.append(((np.copy(elf)+bg)/(4*bg_mult),0, str(4*bg_mult), bg_data_d[j]))
	bg = np.random.poisson(1*bg_mult, size=elf.shape)
	bg_samples.append(((np.copy(elf)+bg),0, f"{1*bg_mult:.2f}", bg_data_d[j]))
	bg = np.random.poisson(2*bg_mult, size=elf.shape)
	bg_samples.append(((np.copy(elf)+bg),0, f"{2*bg_mult:.2f}", bg_data_d[j]))
	bg = np.random.poisson(3*bg_mult, size=elf.shape)
	bg_samples.append(((np.copy(elf)+bg),0, f"{3*bg_mult:.2f}", bg_data_d[j]))
	bg = np.random.poisson(4*bg_mult, size=elf.shape)
	bg_samples.append(((np.copy(elf)+bg),0, f"{4*bg_mult:.2f}", bg_data_d[j]))

	bg_samples[-1][0][bg_samples[-1][0]<0]=0
	bg_samples[-2][0][bg_samples[-2][0]<0]=0
	bg_samples[-3][0][bg_samples[-3][0]<0]=0
	bg_samples[-4][0][bg_samples[-4][0]<0]=0
	

import random
random.seed(0)
random.shuffle(bg_samples)
print(len(elves_samples), len(bg_samples))
#bg_samples = bg_samples[:len(elves_samples)]
print(len(bg_samples))
#elves_samples.extend(bg_samples)

# Count the labels
labels = np.array([el[1] for el in elves_samples])
print("labels", np.count_nonzero(labels==0), np.count_nonzero(labels==1))
print(bg_cnt, elf_cnt)
#exit()

random.shuffle(elves_samples)
# To have some elves in the second half that are not in the first - for validation tests
div_elves_samples1 = elves_samples[:19]+bg_samples[:33]
random.shuffle(div_elves_samples1)
div_elves_samples2 = elves_samples[19:]+bg_samples[33:]
random.shuffle(div_elves_samples2)
elves_samples = div_elves_samples1+div_elves_samples2

# Shuffle the array to mix bg and elves
random.shuffle(elves_samples)

# Add dimension corresponding to channels - needed for CNN
elves_samples = [(np.expand_dims(el[0], 0).astype(np.float32), el[1:]) for el in elves_samples]


import pickle
with open("all_elves_samples.pk", "wb") as f:
	pickle.dump(elves_samples, f)

