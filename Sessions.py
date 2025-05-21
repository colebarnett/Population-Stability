# -*- coding: utf-8 -*-
"""
Created on Fri May 16 13:18:38 2025

@author: coleb
"""
import numpy as np


def get_sessions():
    
    sess = [
            # "braz20220315_07_te90",
            # "braz20220316_06_te96",
            # "braz20220318_07_te108",
            # "braz20220319_04_te112",
            # "braz20220321_04_te116",
            # "braz20220324_19_te176",
            # "braz20220328_04_te206",
            # "braz20220331_04_te215",
            # "braz20220401_05_te220",
            # "braz20220405_04_te235",
            # "braz20220407_04_te243",
            # "braz20220411_04_te257",
            # "braz20220414_04_te286",
            # "braz20220416_04_te294",
            # "braz20220418_05_te299",
            # "braz20220421_04_te312",
            # "braz20220422_04_te316",
            # "braz20220425_04_te320",
            "braz20220426_05_te325",
            "braz20220427_04_te329",
            "braz20220428_04_te333",
            "braz20220429_04_te337",
            "braz20220504_04_te376",
            "braz20220505_04_te380",
            "braz20220507_04_te391",
            "braz20220510_04_te399",
            "braz20220511_04_te408",
            "braz20220512_04_te412",
            "braz20220514_04_te421",
            "braz20220516_04_te425",
            "braz20220517_05_te432",
            "braz20220518_04_te436",
            "braz20220520_04_te445",
            "braz20220607_04_te462",
            "braz20220608_04_te466",
            # "braz20220609_04_te470",
            "braz20220611_04_te478",
            "braz20220613_04_te482",
            "braz20220614_04_te486",
            "braz20220615_04_te490",
            "braz20220617_04_te498",
            "braz20220620_05_te503",
            "braz20220622_04_te511",
            "braz20220623_04_te515",
            "braz20220624_06_te521",
            "braz20220627_04_te529",
            "braz20220629_04_te537"
            ]
    
    all_chs = set(np.arange(128))
    lazy_list_bad_chs = {1,10,12,14,15,28,30,31,32,33,34,35,36,37,46,47,62,63,65,76,78,79}
    
    good_ch_dict = {
            "braz20220315_07_te90": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220316_06_te96": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220318_07_te108": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220319_04_te112": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220321_04_te116": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220324_19_te176": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220328_04_te206": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220331_04_te215": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220401_05_te220": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220405_04_te235": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220407_04_te243": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220411_04_te257": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220414_04_te286": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220416_04_te294": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220418_05_te299": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220421_04_te312": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220422_04_te316": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220425_04_te320": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220426_05_te325": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220427_04_te329": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220428_04_te333": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220429_04_te337": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220504_04_te376": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220505_04_te380": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220507_04_te391": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220510_04_te399": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220511_04_te408": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220512_04_te412": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220514_04_te421": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220516_04_te425": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220517_05_te432": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220518_04_te436": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220520_04_te445": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220607_04_te462": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220608_04_te466": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220609_04_te470": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220611_04_te478": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220613_04_te482": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220614_04_te486": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220615_04_te490": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220617_04_te498": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220620_05_te503": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220622_04_te511": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220623_04_te515": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220624_06_te521": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220627_04_te529": np.array(list(all_chs - lazy_list_bad_chs)),
            "braz20220629_04_te537": np.array(list(all_chs - lazy_list_bad_chs))
            }
    
    return sess, good_ch_dict


def get_good_chs(session):
    
    all_chs = np.arange(128)
    
    
