# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:41:42 2022
Optical Classification and Spectral Scoring System for global waters (OC_3S)

This code is based on the implementation by Wei et al. (2016). Special thanks to the authors for their contribution.

Citation:
Men J, Chen X, Hou X, et al. OC_3S: An optical classification and spectral scoring system for global waters using UV–visible remote sensing reflectance[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2023, 200: 153-172.

"""
import numpy as np
import numpy.matlib as npm
import h5py

def OC_3S_v1(test_Rrs, test_lambda):
    """
    Optical water type classification system based on spectral angle mapping.
    
    Parameters:
        test_Rrs: Remote sensing reflectance data to be classified
        test_lambda: Wavelengths corresponding to the Rrs data
    
    Returns:
        clusterID: Water type classification results (1-30)
        totScore: Classification confidence scores
    """
    
    # Load reference data from HDF5 file containing water type characteristics
    ref_table = h5py.File(r'./Water_classification_system_30c.h5','r')
    up = np.array(ref_table['upB'],'float32')/100000      # Upper bounds for each water type
    low = np.array(ref_table['lowB'],'float32')/100000     # Lower bounds for each water type
    ref = np.array(ref_table['ref_cluster'],'float32')/100000   # Reference cluster spectra
    waves = np.array(ref_table['waves'],'float32')       # Reference wavelengths
    
    # Validate and adjust input wavelength range (380-750nm)
    min_lam = min(test_lambda)
    max_lam = max(test_lambda)
    if min_lam < 380:
        print('Minimum wavelength is %d, it should be >=380'%(min(waves)))
        idx_out = np.argwhere(test_lambda>=380).squeeze()
        test_lambda = test_lambda[idx_out]
        test_Rrs = test_Rrs[:,idx_out]
    if max_lam > 750:
        print('Maximum wavelength is %d, it should be <750'%(max(waves)))
        idx_out = np.argwhere(test_lambda<=750).squeeze()
        test_lambda = test_lambda[idx_out]
        test_Rrs = test_Rrs[:,idx_out]
        
    # Match reference data wavelengths with input wavelengths
    idx = np.array([np.argwhere(waves==test_lambda[i]) for i in range(len(test_lambda))]).squeeze()
    upB = up[:,idx]     # Extract matching upper bounds
    lowB = low[:,idx]   # Extract matching lower bounds
    ref_nRrs = ref[:,idx]  # Extract matching reference spectra
    
    # Validate input data dimensions
    if test_lambda.ndim > 1:
        row_lam, len_lam = test_lambda.shape
        if row_lam != 1:
            test_lambda = np.transpose(test_lambda)
            row_lam, len_lam = test_lambda.shape
    else:
        row_lam = 1
        len_lam = len(test_lambda)

    # Check if Rrs and wavelength dimensions match
    row, col = test_Rrs.shape
    if len_lam != col and len_lam != row:
        print('Rrs and lambda size mismatch, please check the input data!')
    elif len_lam == row:
        test_Rrs = np.transpose(test_Rrs)

    # Get dimensions of reference spectra
    refRow, _ = ref_nRrs.shape

    # Store original Rrs values
    test_Rrs_orig = test_Rrs
    
    # Normalize input spectra
    inRow, inCol = np.shape(test_Rrs)
    test_Rrs = np.transpose(test_Rrs)
    test_Rrs_orig = np.transpose(test_Rrs_orig)
    
    # Calculate normalization denominator
    nRrs_denom = np.sqrt(np.nansum(test_Rrs**2, 0))
    nRrs_denom = npm.repmat(nRrs_denom, inCol, 1)
    nRrs = test_Rrs/nRrs_denom
    
    # Prepare arrays for Spectral Angle Mapping (SAM)
    test_Rrs2 = np.repeat(test_Rrs_orig[:, :, np.newaxis], refRow, axis=2)
    nRrs2_denom = np.sqrt(np.nansum(test_Rrs2**2, 0))
    nRrs2_denom = np.repeat(nRrs2_denom[:,:, np.newaxis], inCol, axis=2)
    nRrs2_denom = np.moveaxis(nRrs2_denom, 2, 0)
    nRrs2 = test_Rrs2/nRrs2_denom
    nRrs2 = np.moveaxis(nRrs2, 2, 1)

    # Normalize reference spectra
    ref_nRrs = np.transpose(ref_nRrs)
    ref_nRrs2 = np.repeat(ref_nRrs[:,:, np.newaxis], inRow, axis=2)
    ref_nRrs2_denom = np.sqrt(np.nansum(ref_nRrs2**2, 0))
    ref_nRrs2_denom = np.repeat(ref_nRrs2_denom[:,:, np.newaxis], inCol, axis=2)
    ref_nRrs2_denom = np.moveaxis(ref_nRrs2_denom, 2, 0)
    ref_nRrs_corr2 = ref_nRrs2/ref_nRrs2_denom

    # Perform Spectral Angle Mapping classification
    cos_denom = np.sqrt(np.nansum(ref_nRrs_corr2**2, 0) * np.nansum(nRrs2**2, 0))
    cos_denom = np.repeat(cos_denom[:, :, np.newaxis], inCol, axis=2)
    cos_denom = np.moveaxis(cos_denom, 2, 0)
    cos = (ref_nRrs_corr2*nRrs2)/cos_denom
    cos = np.nansum(cos, 0)
    
    # Find best matching water type
    maxCos = np.amax(cos, axis=0)  # Maximum cosine similarity
    clusterID = np.argmax(cos, axis=0)  # Water type with highest similarity
    posClusterID = np.isnan(maxCos)  # Track invalid classifications
    
    # Calculate classification confidence scores
    upB_corr = np.transpose(upB) 
    lowB_corr = np.transpose(lowB)
    
    # Apply tolerance bounds
    upB_corr2 = upB_corr[:,clusterID] * (1+0.01)  # Add 1% tolerance to upper bound
    lowB_corr2 = lowB_corr[:,clusterID] * (1-0.01)  # Subtract 1% tolerance from lower bound
    ref_nRrs2 = ref_nRrs[:,clusterID]

    # Normalize bounds
    ref_nRrs2_denom = np.sqrt(np.nansum(ref_nRrs2**2, 0))
    ref_nRrs2_denom = np.transpose(np.repeat(ref_nRrs2_denom[:,np.newaxis], inCol, axis=1))
    upB_corr2 = upB_corr2 / ref_nRrs2_denom
    lowB_corr2 = lowB_corr2 / ref_nRrs2_denom

    # Calculate differences from bounds
    upB_diff = upB_corr2 - nRrs
    lowB_diff = nRrs - lowB_corr2

    # Calculate confidence scores
    C = np.empty([inCol,inRow], dtype='float')*0
    pos = np.logical_and(upB_diff>=0, lowB_diff>=0)  # Within bounds
    C[pos] = 1  # Assign score of 1 to spectra within bounds
    C[:,posClusterID] = np.nan  # Mark invalid classifications                                              
    totScore = np.nanmean(C, 0)  # Calculate mean confidence score
    
    # Prepare final results
    clusterID = clusterID.astype('float')
    clusterID[posClusterID] = np.nan
    clusterID = clusterID + 1  # Convert from 0-based to 1-based indexing

    return clusterID, totScore
