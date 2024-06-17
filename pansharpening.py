import numpy as np
import h5py as h5
import hypertools as hyp


#############################################################################################
################################## PCA Pansharpening ########################################
#############################################################################################


def PCA(X, no_dims):

    """
    Perform Principal Component Analysis (PCA) on input data X.

    Args:
        X (array-like): Input data matrix of shape (n_samples, n_features).
        no_dims (int): Number of principal components to retain.

    Returns:
        mappedX (array-like): Transformed data matrix after PCA of shape (n_samples, no_dims).
        mapping (dict): Dictionary containing PCA mapping information.
            - 'mean': Mean of elements for each band.
            - 'M': Principal axes in feature space, representing the directions of maximum variance.
            - 'lambda': Eigenvalues corresponding to the principal components.
    """

    print('PCA start')
    mapping = {}
    # Calculate mean of elements for each band
    print('Calculate mean of elements for each band')
    mapping['mean'] = np.ma.mean(X, axis=0)
    # Subtract the column-wise zero empirical mean
    X = X - mapping['mean']
    # Calculate the covariance matrix
    C = np.ma.cov(X, rowvar=False)
    # Perform eigenvalue decomposition
    eigenvalues, M = np.linalg.eigh(C, UPLO='U')
    # Sort eigenvalues and corresponding eigenvectors in descending order
    ind = np.arange(0, eigenvalues.shape[0], 1)
    ind = np.flip(ind)
    M = M[:, ind[0:no_dims]]
    eigenvalues = eigenvalues[0:no_dims]
    # Project the data onto the new basis
    mappedX = np.ma.dot(X, M)
    # Store the principal axes, eigenvalues, and mean in the mapping dictionary
    mapping['M'] = M
    mapping['lambda'] = eigenvalues
    print('PCA end')
    return (mappedX, mapping)


def inversePCA(E, P, MeanV):

    """
    Perform inverse Principal Component Analysis (PCA) to reconstruct the original data matrix.

    Args:
        E (array-like): Principal axes in feature space (M), shape (n_features, n_components).
        P (array-like): Transformed data matrix after PCA (mappedX), shape (n_samples, n_components).
        MeanV (array-like): Mean of elements for each band.

    Returns:
        reconstructed (array-like): Reconstructed data matrix, shape (n_samples, n_features).
    """

    # Perform inverse PCA by multiplying mapped data by the transpose of principal axes and adding the mean
    reconstructed = np.ma.dot(P, E.T) + MeanV
    return reconstructed


def pca_pansharpening(hyper, panchro, name_file_to_save='PCA_pansharpening.h5'):

    """
    Applies PCA pansharpening to a hyperspectral image using a panchromatic image

    Args:
        hyper (Array): Hyperspectral image [x, y, bands]
        panchro (Array): Panchromatic image [x, y]
        name_file_to_save (str): Name of the file to save the pansharpened image (default is 'PCA_pansharpening.h5')
        
    Returns:
        pansharpened (Array): Pansharpened hyperspectral image [x, y, bands]
    """

    # Check if rescaling is needed and rescale if necessary
    if hyper.shape != (panchro.shape[0], panchro.shape[1], hyper.shape[2]):
        hyper = hyp.rescaling(panchro, hyper)
    print('>>PCA pansharpening started<<')
    # Reshape the hyperspectral image for PCA
    m, n, d = hyper.shape
    M = np.reshape(hyper, (m * n, d))
    # Perform PCA
    PCAData, PCAMap = PCA(M, d)
    PCAData = np.reshape(PCAData, (m, n, d))
    F = PCAData
    # Adjust the first principal component with the panchromatic image
    PC1 = (panchro - panchro.mean()) * (F[:, :, 0].std() / panchro.std()) + F[:, :, 0].mean()
    F[:, :, 0] = PC1
    print('Inverse PCA')
    # Perform inverse PCA
    F = inversePCA(PCAMap['M'], np.reshape(F, (m * n, d)), PCAMap['mean'])
    pansharpened = np.reshape(F, (m, n, d))    
    # Clip values to the valid range and convert to uint16
    pansharpened = np.clip(pansharpened, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    print('Saving file')
    # Save the pansharpened image to an HDF5 file
    with h5.File(name_file_to_save, 'w') as f:
        f.create_dataset('PCA_pansharpening', data=pansharpened)
    print('>>PCA pansharpening end<<')
    return pansharpened


#############################################################################################
################################# Brovey pansharpening ######################################
#############################################################################################


def brovey_pansharpening(hyper, panchro, name_file_to_save='brovey_pansharpening.h5'):

    """
    Applies Brovey transformation-based pansharpening to a hyperspectral image using a panchromatic image

    Args:
        hyper (Array): Hyperspectral image [x, y, bands]
        panchro (Array): Panchromatic image [x, y]
        name_file_to_save (str): Name of the file to save the pansharpened image (default is 'brovey_pansharpening.h5')
        
    Returns:
        pansharpened (Array): Pansharpened hyperspectral image [x, y, bands]
    """

    # Check if rescaling is needed and rescale if necessary
    if hyper.shape != (panchro.shape[0], panchro.shape[1], hyper.shape[2]):
        hyper = hyp.rescaling(panchro, hyper)
    hyper_float64 = hyper.astype(np.float64)
    # Obtain the average sum matrix of all channels
    hyper_average_sum = np.true_divide(np.sum(hyper_float64, axis=2), hyper_float64.shape[2])  
    # Obtain the ratio between panchromatic and hyper_sum
    ratio = np.true_divide(panchro.astype(np.float64), hyper_average_sum)
    # Multiply the ratio to all channels of the hyper image
    pansharpened = np.einsum('abc,ab->abc', hyper, ratio)
    # Spectral richness adjustment
    max_val, min_val = np.max(pansharpened), np.min(pansharpened)
    pansharpened = ((pansharpened - min_val) * np.iinfo(np.uint16).max) / (max_val - min_val)
    # Apply clip and data type uint16 to pansharpened
    pansharpened = np.clip(pansharpened, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    # Save the pansharpened image to an HDF5 file
    with h5.File(name_file_to_save, 'w') as f:
        f.create_dataset('brovey_pansharpening', data=pansharpened)    
    return pansharpened
