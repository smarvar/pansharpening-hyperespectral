# Pansharpening-Hyperspectral

## Panchromatic Refinement: A Strategy for Increasing Resolution in Hyperspectral Satellite Imagery

This notebook contains all the functionalities to apply the pansharpening technique to hyperspectral or multispectral images from a high resolution panchromatic image. The next pansharpening techniques were implemented:   

- **PCA pansharpenig**
- **Brovey Transform**

Both methods allow the spatial resolution of the **hyperspectral/multispectral** images to be increased to the resolution of the associated **panchromatic** image. 
Different metrics were implemented to evaluate the quality of the panchromatic fusion obtained by both methods. These metrics include spatial correlation index (Q), spectral correlation index (SCC), mean squared error (MSE), signal to noise ratio (SNR). 
Also, different visual comparisons were generated, based on the corresponding images and histograms of the hyperspectral set and the results of the pansharpening methods. 

## Methodology diagram
![methodology](https://github.com/smarvar/pansharpening-hyperespectral/blob/main/pansharpening_methodology.png)

## Notebook contents:

- Hyperspectral data
- Preprocesing 
- Pansharpening methods: PCA, Brovey
- Visual comparison
- Validation metrics
- Histogram comparison

This repository contains the following files: `pansharpening_notebook.ipynb` with all the code execution, `pansharpening.py` with the implemented pansharpening methods, `hypermetrics.py` with the different evaluation metrics and `hypertools.py` with auxiliary and visualisation functions. 


## Hyperspectral data (HSI)

Hyperspectral images from the **PRISMA** satellite were used in this study. To access the PRISMA data you can visit the following link:  https://prismauserregistration.asi.it/

Once you download the PRISMA hyperspectral image you will have an HDF5 file, the hyp.open_pansharpened_file function allows you to access the information in the file.  

Note: if you already have the hyperspectral or multispectral data with the panchromatic image from another satellite, you must view the data format and generate a hyper[i,j,n] matrix where n corresponds to the spectral bands.  In addition, a PANCHRO[I,J] matrix corresponding to the panchromatic image must be generated. Once this is done, you can move on to the preprocessing step. 

## Results:
The results presented correspond to a PRISMA hyperspectral capture over the municipality of Punta Alta, northeast of the Bahía Blanca Estuary, province of Buenos Aires, Argentina. 

The image shows a comparison between the panchromatic image, a hyperspectral band and the result of the PCA pansharpening method.
- Pancrhomatic shape: (6000, 6000) 
- Hyperspectral shape: (1000, 1000, 63)
- PCA pansharpened shape: (6000, 6000, 63)

![hyper_comparasion](https://github.com/smarvar/pansharpening-hyperespectral/blob/main/panchromatic_hyperspectral_PCA-pansharpened.png)

The following image shows a comparison between the panchromatic image, a hyperspectral band and the result of the PCA pansharpening method, zooming in on a 500 pixel region in the image. 

![hyper_comparasion_zoomed](https://github.com/smarvar/pansharpening-hyperespectral/blob/main/panchromatic_hyperspectral_PCA-pansharpened_zoomed.png)

