# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:41:29 2023

@author: menjilin
"""
from OC_3S import OC_3S_v1
import numpy as np

if __name__=='__main__':
    # Remote sensing reflectance 
    test_Rrs = np.array([[0.00695552,0.00611691,0.00544757,0.00452322,0.00220445,0.000148625,0.00024075],
                        [0.0073947,0.00652504,0.0059856,0.00500528,0.00243824,0.000217908,0.000318192],
                        [0.00512031,0.00455387,0.004265,0.00350278,0.00174399,0.000147439,0.00030923],
                        [0.00670229,0.00577306,0.00522317,0.00427592,0.00199355,0.000118097,0.000149938],
                        [0.00623312,0.00538205,0.00496898,0.00413726,0.00194883,0.000153799,0.000275778],
                        [0.00582792,0.00498743,0.00439933,0.00360783,0.00161173,0.000103841,0.00013843],
                        [0.014652,0.0140562,0.0162978,0.0177578,0.0240399,0.0146119,0.0146733],
                        [0.00883352,0.00808966,0.00832895,0.00873099,0.012089,0.00735695,0.00800812]])
    # wavelength
    test_lambda = np.array([412,443,490,510,555,667,680])
    
    #implement OC_3S
    [WC, Score] = OC_3S_v1(test_Rrs, test_lambda)  #WC: water class
    
    for wc, score in zip(WC, Score):
        print("Water type: %d, score: %.2f" % (wc, score))
