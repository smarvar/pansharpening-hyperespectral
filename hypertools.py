import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from skimage.transform import rescale
from tqdm import tqdm


#############################################################################################
######################################### Tools #############################################
#############################################################################################


def rescaling(im_reference, im_to_rescale, save=False):

    """
    Rescales im_to_rescale to match the dimensions of im_reference

    Args:
        im_reference (Array): Reference hyperspectral image [x, y, bands]
        im_to_rescale (Array): Hyperspectral image to be rescaled [x, y, bands]
        save (bool): Whether to save the rescaled image to a file (default is False)
        
    Returns:
        im_rescaled (Array): Rescaled hyperspectral image matching the dimensions of im_reference [x, y, bands]
    """
    print('Rescaling to reference image')
    scale_x = im_reference.shape[0] / im_to_rescale.shape[0]
    scale_y = im_reference.shape[1] / im_to_rescale.shape[1]
    n_bands = im_to_rescale.shape[2]
    
    # Initialize an array for the rescaled image
    im_rescaled = np.zeros((im_reference.shape[0], 
                            im_reference.shape[1], 
                            im_to_rescale.shape[2]), 
                           dtype=np.uint16)

    # Rescale each band of the image
    for i in tqdm(range(n_bands)):
        im_rescaled[:,:,i] = rescale(im_to_rescale[:,:,i], scale=(scale_x, scale_y), 
                                     mode='constant', order=3, anti_aliasing=False, 
                                     clip=True, preserve_range=True).astype(np.uint16)
        
    print('Hyper Shape before: {}, after: {}'.format(im_to_rescale.shape, im_rescaled.shape))    

    # Save the rescaled image if requested
    if save: 
        with h5.File('hyper_up.h5', 'w') as f:
            f.create_dataset('hyper_up', data = im_rescaled)    
            
    return im_rescaled


def open_pansharpened_file(root_file=''):
    
    """
    Opens an HDF5 file containing pansharpened images and lists its elements

    Args:
        root_file (str): Path to the HDF5 file
        
    Returns:
        file: The opened HDF5 file object
    """
    # Open the HDF5 file
    file = h5.File(str(root_file), 'r')
    # Get the list of elements in the file
    file_list = list(file.keys()) 
    # Print the elements in the file
    print('Elements in file: {}'.format(file_list))
    return file


def spectral_adjustment(hyper):

    """
    Adjusts the spectral range of a hyperspectral image to the full range of uint16

    Args:
        hyper (Array): Hyperspectral image [x, y, bands]
        
    Returns:
        hyper_adjusted (Array): Spectrally adjusted hyperspectral image [x, y, bands]
    """
    # Find the maximum and minimum values in the hyperspectral image
    max_val, min_val = np.max(hyper), np.min(hyper)
    # Adjust the spectral range to the full range of uint16
    hyper_adjusted = ((hyper - min_val) * np.iinfo(np.uint16).max) / (max_val - min_val)
    return hyper_adjusted


def snr_finding_null_values(im, snr_ref=0):

    """
    Finds null bands in a hyperspectral image based on a reference SNR threshold

    Args:
        im (Array): Hyperspectral image [x, y, bands]
        snr_ref (float): Reference SNR threshold, default is 0
        
    Returns:
        snr_db (Array): SNR values in dB for each band
        null_bands (Array): Indices of bands with SNR below the reference threshold
    """
    # Calculate the signal as the mean across spatial dimensions
    signal = np.mean(im, axis=(0, 1))
    # Calculate the noise as the standard deviation across spatial dimensions
    noise = np.std(im, axis=(0, 1))
    # Calculate SNR, handle division by zero
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise != 0)
    # Convert SNR to dB, handle log of zero
    snr_db = 10 * np.log10(snr, out=np.zeros_like(snr), where=snr != 0)
    # Count the number of bands with SNR below the reference threshold
    count = np.sum(snr_db <= snr_ref)
    # Find the indices of bands with SNR below the reference threshold
    null_bands = np.where(snr_db <= snr_ref)[0]
    # Print the total number of null bands found and their indices
    if count > 0: 
        print('Total null bands found: {}, Null band number: {}'.format(count, null_bands))
    return snr_db, null_bands


def remove_bands(hyp, n_bands_array):
    return np.delete(hyp, n_bands_array, axis=2)


#############################################################################################
####################################### Graphics ############################################
#############################################################################################


def info_graph(im, titlte=''):

    print("Shape: {} and Type: {}".format(im.shape, im.dtype))
    print('Pixel range: [{}, {}]'.format(np.min(im), np.max(im)))
    vmin, vmax = 0, im.max()
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
    axs.set_title(titlte)
    plt.show()


def compare_modified_imgs(im_before, im_after, band, title_im_modified=''):

    print("Image Before: Shape {} and Type {}".format(im_before.shape, im_before.dtype))
    print("Image After: Shape {} and Type {}".format(im_after.shape, im_after.dtype))
    print('Minimum pixel value: Before {}, After {}'.format(np.min(im_before[:,:,band]), np.min(im_after[:,:,band])))
    print('Maximum pixel value: Before {}, After  {}'.format(np.max(im_before[:,:,band]), np.max(im_after[:,:,band])))   
    v = np.array([np.max(im_before), np.max(im_after)])  
    vmin, vmax = 0, v.max()
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    axs[0].imshow(im_before[:,:,band], cmap='gray', vmin=vmin, vmax=vmax)
    axs[0].set_title('Image Before {}'.format(title_im_modified))
    axs[1].imshow(im_after[:,:,band], cmap='gray', vmin=vmin, vmax=vmax)
    axs[1].set_title('Image After {}'.format(title_im_modified))
    plt.show()


def triple_graph(hyp_0, hyp_1, hyp_2, title_0='', title_1='', title_2=''):
    
    v = np.array([np.max(hyp_0), np.max(hyp_1), np.max(hyp_2)])  
    vmin, vmax = 0, v.max()
    fig, axs = plt.subplots(1, 3, figsize=(20, 20))
    axs[0].imshow(hyp_0, cmap='gray')
    axs[0].set_title('{}'.format(title_0) )
    axs[1].imshow(hyp_1, cmap='gray',  vmin=vmin, vmax=vmax)
    axs[1].set_title('{}'.format(title_1))
    axs[2].imshow(hyp_2, cmap='gray',  vmin=vmin, vmax=vmax)
    axs[2].set_title('{}'.format(title_2))
    plt.show()


def triple_zoom_graph(hyp_0, hyp_1, hyp_2, cx, cy, npixels, title_0='', title_1='', title_2=''):
    
    cy2 = cy+npixels
    cx2 = cx+npixels
    scale = hyp_0.shape[0]/hyp_1.shape[0]
    v = np.array([np.max(hyp_0), np.max(hyp_1), np.max(hyp_2)])  
    vmin, vmax = 0, v.max()
    fig, axs = plt.subplots(1, 3, figsize=(20, 20))
    axs[0].imshow(hyp_0[cy:cy2, cx:cx2], cmap='gray')
    axs[0].set_title('{}'.format(title_0) )
    axs[1].imshow(hyp_1[int(cy/scale):int(cy2/scale), 
                        int(cx/scale):int(cx2/scale)], 
                        cmap='gray', vmin=vmin, vmax=vmax)
    axs[1].set_title('{}'.format(title_1))
    axs[2].imshow(hyp_2[cy:cy2, cx:cx2], cmap='gray', vmin=vmin, vmax=vmax)
    axs[2].set_title('{}'.format(title_2))
    plt.show()


def hyper_histogram(hyp_1, hyp_2, n_bins, title_1='', title_2=''):

    h1 = hyp_1[:,:].ravel()
    h2 = hyp_2[:,:].ravel()
    _, nbins = np.histogram(h1, bins=n_bins, range=[0, 65536])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(h1, bins=nbins, edgecolor='black')
    axs[0].set_title('Hyper histogram {}'.format(title_1))
    axs[1].hist(h2, bins=nbins, edgecolor='black')
    axs[1].set_title('Hyper histogram {}'.format(title_2))
    plt.show()


def plot_superposed_histograms(images, images_name, normalised=False, num_bins=100):

    plt.figure(figsize=(8, 6), dpi=120)

    for i, image in enumerate(images):
        _ , bins = np.histogram(image.flatten(), bins=num_bins)
        plt.hist(image.ravel(), bins=bins, label=images_name[i], 
                 density=normalised, histtype='step', stacked=True, fill=False)

    plt.xlabel('Radiance (DN)', fontsize=14)
    plt.ylabel('Pixel count', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim([0, 15000])
    plt.legend()
    plt.grid(True)
    plt.show()
