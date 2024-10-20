# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:41:42 2022
Optical Classification and Spectral Scoring System for global waters (OC_3S)
"OC_3S: An optical classification and spectral scoring system for global waters using UV–visible remote sensing reflectance"
"""
import numpy as np
import numpy.matlib as npm
import h5py

def OC_3S_v1(test_Rrs, test_lambda):
    '''load OC_3S file'''
    ref_table = h5py.File(r'./Water_classification_system_30c.h5','r')
    up = np.array(ref_table['upB'],'float32')
    low = np.array(ref_table['lowB'],'float32')
    ref = np.array(ref_table['ref_cluster'],'float32')
    waves = np.array(ref_table['waves'],'float32')
    
    min_lam = min(test_lambda)
    max_lam = max(test_lambda)
    if min_lam<380:
        print('Minimum wavelength is %d, it should be >=380'%(min(waves)))
        idx_out = np.argwhere(test_lambda>=380).squeeze()
        test_lambda = test_lambda[idx_out]
        test_Rrs = test_Rrs[:,idx_out]
    if max_lam>750:
        print('Maximum wavelength is %d, it should be <750'%(max(waves)))
        idx_out = np.argwhere(test_lambda<=750).squeeze()
        test_lambda = test_lambda[idx_out]
        test_Rrs = test_Rrs[:,idx_out]
    '''find ref data corresponding to test_lambda'''
    idx = np.array([np.argwhere(waves==test_lambda[i]) for i in range(len(test_lambda))]).squeeze()
    upB = up[:,idx]
    lowB = low[:,idx]
    ref_nRrs = ref[:,idx]
    
    ''' Check input data '''
    if test_lambda.ndim > 1:
        row_lam, len_lam = test_lambda.shape
        if row_lam != 1:
            test_lambda = np.transpose(test_lambda)#np.transpose
            row_lam, len_lam = test_lambda.shape
    else:
        row_lam = 1
        len_lam = len(test_lambda)

    row, col = test_Rrs.shape
    if len_lam != col and len_lam != row:
        print('Rrs and lambda size mismatch, please check the input data!')
    elif len_lam == row:
        test_Rrs = np.transpose(test_Rrs)

    ''' 30 Normalized spectral water types '''  
    refRow, _ = ref_nRrs.shape

    ''' Match the ref_nRrs and test_Rrs '''
    # keep the original value
    test_Rrs_orig = test_Rrs
    
    ''' Normalization '''
    inRow, inCol = np.shape(test_Rrs)

    # transform spectrum to column, inCol*inRow
    test_Rrs = np.transpose(test_Rrs)
    test_Rrs_orig = np.transpose(test_Rrs_orig)

    # inCol*inRow
    nRrs_denom = np.sqrt(np.nansum(test_Rrs**2, 0))
    # nRrs_denom = repmat(nRrs_denom,[inCol,1]);
    nRrs_denom = npm.repmat(nRrs_denom, inCol, 1)
    nRrs = test_Rrs/nRrs_denom;      

    # SAM input, inCol*inRow*refRow 
    test_Rrs2 = np.repeat(test_Rrs_orig[:, :, np.newaxis], refRow, axis=2)

    # #for ref Rrs, inCol*refRow*inRow 
    # test_Rrs2p = np.moveaxis(test_Rrs2, 2, 1)

    # inCol*inRow*refRow  
    nRrs2_denom = np.sqrt(np.nansum(test_Rrs2**2, 0))
    # nRrs2_denom = repeat(nRrs2_denom, inCol, axis=2)
    nRrs2_denom = np.repeat(nRrs2_denom[:,:, np.newaxis], inCol, axis=2)
    nRrs2_denom = np.moveaxis(nRrs2_denom, 2, 0)
    nRrs2 = test_Rrs2/nRrs2_denom
    # inCol*refRow*inRow  
    nRrs2 = np.moveaxis(nRrs2, 2, 1)

    ''' Adjust the ref_nRrs, according to the matched wavebands '''
    #row,_  = ref_nRrs.shape

    #### re-normalize the ref_adjusted
    ref_nRrs = np.transpose(ref_nRrs)

    # inCol*refRow*inRow 
    ref_nRrs2 = np.repeat(ref_nRrs[:,:, np.newaxis], inRow, axis=2)

    # inCol*refRow*inRow 
    ref_nRrs2_denom = np.sqrt(np.nansum(ref_nRrs2**2, 0))
    ref_nRrs2_denom = np.repeat(ref_nRrs2_denom[:,:, np.newaxis], inCol, axis=2)
    ref_nRrs2_denom = np.moveaxis(ref_nRrs2_denom, 2, 0)
    ref_nRrs_corr2 = ref_nRrs2/ref_nRrs2_denom

    ''' Classification '''
    #### calculate the Spectral angle mapper
    # inCol*refRow*inRow 
    cos_denom = np.sqrt(np.nansum(ref_nRrs_corr2**2, 0) * np.nansum(nRrs2**2, 0))
    cos_denom = np.repeat(cos_denom[:, :, np.newaxis], inCol, axis=2)
    cos_denom = np.moveaxis(cos_denom, 2, 0)
    cos = (ref_nRrs_corr2*nRrs2)/cos_denom
    # refRow*inRow 
    cos = np.nansum(cos, 0)
    
    # 1*inRow
    maxCos = np.amax(cos, axis=0) 
    clusterID = np.argmax(cos, axis=0) # finds location of max along an axis, returns int64
    posClusterID = np.isnan(maxCos)

    ''' Scoring '''
    upB_corr = np.transpose(upB) 
    lowB_corr = np.transpose(lowB)

    ''' Comparison '''
    # inCol*inRow
    upB_corr2 = upB_corr[:,clusterID] * (1+0.01)
    lowB_corr2 = lowB_corr[:,clusterID] * (1-0.01)
    ref_nRrs2 = ref_nRrs[:,clusterID]

    #normalization
    ref_nRrs2_denom = np.sqrt(np.nansum(ref_nRrs2**2, 0))
    ref_nRrs2_denom = np.transpose(np.repeat(ref_nRrs2_denom[:,np.newaxis], inCol, axis=1))
    upB_corr2 = upB_corr2 / ref_nRrs2_denom
    lowB_corr2 = lowB_corr2 / ref_nRrs2_denom

    upB_diff = upB_corr2 - nRrs
    lowB_diff = nRrs - lowB_corr2

    C = np.empty([inCol,inRow], dtype='float')*0
    pos = np.logical_and(upB_diff>=0, lowB_diff>=0)
    C[pos] = 1

    #process all NaN spectral 
    C[:,posClusterID] = np.nan                                               

    totScore = np.nanmean(C, 0)
    clusterID = clusterID.astype('float')
    clusterID[posClusterID] = np.nan
    # Convert from index to water type 1-30
    clusterID = clusterID +1

    return clusterID, totScore