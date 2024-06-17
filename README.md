# Pansharpening-Hyperspectral

## Panchromatic Refinement: A Strategy for Increasing Resolution in Hyperspectral Satellite Imagery

By: 

PhD(c) Steven Martinez Vargas
mail: smarvar@outlook.com

This notebook contains all the functionalities to apply the pansharpening technique to hyperspectral or multispectral images from a high resolution panchromatic image.  The next pansharpening techniques were implemented:   
- PCA pansharpenig
- Brovey Transform

Both methods allow the spatial resolution of the hyperspectral/multispectral images to be increased to the resolution of the associated panchromatic image. 

Different metrics were implemented to evaluate the quality of the panchromatic fusion obtained by both methods. These metrics include spatial correlation index (Q), spectral correlation index (SCC), mean squared error (MSE), signal to noise ratio (SNR). 

Different visual comparisons were generated, based on the corresponding images and histograms of the hyperspectral set and the results of the pansharpening methods. 

## Notebook contents:

- Hyperspectral data
- Preprocesing 
- Pansharpening methods: PCA, Brovey
- Visual comparison
- Validation metrics
- Histogram comparison
