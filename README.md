# MARSIS-xDR-READER
The Python tool presented here is capable of reading common data types used to distribute MARSIS dataset and then converting into multiple data formats.

Users can interactively configure data source, destination, pre-processing and type of outputs among:
* Geopackages: used in GIS software, is a single self-contained file containing a layer in which are stored all parameters for each file processed.
* Numpy array dump: used for fast reading and analysis, containing original data for both frequency.
* PNG images: used for fast inspections, created for each frequency, and saved. Image pre-processing filters, such as:
    * Image-denoising,
    * Standardization
    * Normalization.
* SEG-Y: used for analysing data with seismic interpretation and processing software, see e.g. OpendTect, consist of a SEG-Y for each frequency.
