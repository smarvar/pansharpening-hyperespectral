import numpy as np


def calculate_moran_index(image1, image2):

    # Calculate global means
    mean_image1 = np.mean(image1)
    mean_image2 = np.mean(image2)
    # Subtract global means
    image1_diff = image1 - mean_image1
    image2_diff = image2 - mean_image2
    # Calculate Moran numerator
    moran_numerator = np.sum(image1_diff * image2_diff)
    # Calculate Moran denominator
    moran_denominator = np.sqrt(np.sum(image1_diff**2) * np.sum(image2_diff**2))
    # Calculate Moran Index
    moran_index = moran_numerator / moran_denominator
    return moran_index


def calculate_Q(panchromatic, pansharpened):

    """
    Calculates the spatial correlation index (Q) 
    between panchromatic and hyperspectral. 
    
    Args:
        panchromatic (Array): panchromatic image [x, y]
        pansharpened (Array): Hyperspectral set [x, y, bands]
       
    Returns:
       Q: Spatial correlation index     
    """
    
    q = 0
    n_bands = pansharpened.shape[2]
    for band in range(n_bands):
        pansharp = pansharpened[:,:,band]
        q += calculate_moran_index(panchromatic, pansharp) / n_bands
    return np.round(q, 3)


def normalize_image(image):

    """
    Normalizes the hyperspectral/multispectral image set
    
    Args:
        image (Array): Hyperspectral set [x, y, bands]
       
    Returns:
        normalized_image (Array): Normalized image set [x, y, bands]      
    """
    # Flatten the image to 2D (pixels x bands)
    flat_image = image.reshape(-1, image.shape[2])
    # Normalize each band of the image
    mean = np.mean(flat_image, axis=0)
    std = np.std(flat_image, axis=0)
    normalized_image = (flat_image - mean) / std
    # Reshape to the original shape
    return normalized_image.reshape(image.shape)


def calculate_SCC(spectral_image, pansharpened_image):

    """
    Generates the Spectral Correlation Coefficient (SCC) index between two images,
    using the spectrum of spectral_image as a reference

    Args:
        spectral_image (Array): Reference hyperspectral set [x, y, bands]
        pansharpened_image (Array): Pansharpened hyperspectral set [x, y, bands]
        
    Returns:
        scc: Spectral correlation coefficient with the range (-1, 1) where
            1 values close to 1 indicate a high correlation index
            0 indicates no significant linear correlation between spectral and pansharpened images
           -1 This could suggest that the bands are inverted or there is a significant problem with the pansharpening process
    """
    # Normalize both sets of images
    spectral_nor = normalize_image(spectral_image)
    pansharpened_nor = normalize_image(pansharpened_image)
    # Reshape to 2 dimensions
    spectral_image_2d = spectral_nor.reshape(-1, spectral_nor.shape[2])
    pansharp_2d = pansharpened_nor.reshape(-1, pansharpened_nor.shape[2])
    # Concatenate the two matrices
    combined = np.concatenate((spectral_image_2d, pansharp_2d), axis=1)
    # Calculate the correlation between both images
    corr_matrix = np.corrcoef(combined, rowvar=False)
    # Extract the relevant part of the correlation matrix
    num_bands = spectral_image.shape[2]
    scc_values = corr_matrix[:num_bands, num_bands:]
    # Calculate the self-correlation of the spectral image
    scc_spectral_im = np.corrcoef(spectral_image_2d, rowvar=False)
    # Calculate the ratio between the average of the two correlations
    scc = np.mean(scc_values) / np.mean(scc_spectral_im)
    return np.round(scc, 3)


def calculate_MSE(spectral_image, pansharpened_image):
    
    """
    Calculates the Mean Squared Error (MSE) between two images,
    using spectral_image as a reference

    Args:
        spectral_image (Array): Reference hyperspectral set [x, y, bands]
        pansharpened_image (Array): Pansharpened hyperspectral set [x, y, bands]
        
    Returns:
        mse: Mean Squared Error, where values closer to zero
             indicate that the images are similar
    """
    # Reshape to 2 dimensions
    spectral_image_2d = spectral_image.reshape(-1, spectral_image.shape[2])
    pansharp_2d = pansharpened_image.reshape(-1, pansharpened_image.shape[2])
    # Calculate mse
    mse = np.mean((spectral_image_2d - pansharp_2d)**2)
    return np.round(mse, 3)


def calculate_SNR(spectral_image, pansharpened_image):
    
    """
    Calculates the Signal-to-Noise Ratio (SNR) between two images,
    using spectral_image as a reference

    Args:
        spectral_image (Array): Reference hyperspectral set [x, y, bands]
        pansharpened_image (Array): Pansharpened hyperspectral set [x, y, bands]
        
    Returns:
        snr: Signal-to-Noise Ratio, where higher values indicate a better SNR
    """
    # Normalize both sets of images
    spectral_nor = normalize_image(spectral_image)
    pansharpened_nor = normalize_image(pansharpened_image)
    # Reshape to 2 dimensions
    spectral_image_2d = spectral_nor.reshape(-1, spectral_nor.shape[2])
    pansharp_2d = pansharpened_nor.reshape(-1, pansharpened_nor.shape[2])
    # Calculate snr
    snr = 10 * np.log10(np.var(spectral_image_2d) / np.var(pansharp_2d - spectral_image_2d))
    return np.round(snr, 3)
