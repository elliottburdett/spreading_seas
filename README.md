# spreading_seas

This repository contains an observational data-driven pipeline for selecting member stars in the AAU Stream.
This should be generalized to characterize other streams, given a stream rotation matrix and data catalogs.

<<<<<<< HEAD
-select_spatial_box.py: Reads in a catalog and reduces it to the AAU spatial region.
-rotation_matrix.py: Conducts a coordinate transformation to the stream frame.
-gaussian_membership_models.py: With previously confirmed member stars, fit stream parameters to phi1, and construct likelihood models based on the function.
-select_bhb_rrl.py: Makes a selection of standard candles in the stream region and fits a distance gradient to the stream.
-delve_dr3_map_partitions.py: With the assumed distance gradient, creates an approximated absolute magnitude space and applies an isochrone-based matched filter to select members.
=======
- Select AAU stars: Reads in a catalog and reduces it to the AAU spatial region.
- Rotation matrix: Conducts a coordinate transformation to the stream frame.
- Gaussian membership models: With previously confirmed member stars, fit stream parameters to phi1, and construct likelihood models based on the function.
- BHB and RRL selection: Makes a selection of standard candles in the stream region and fits a distance gradient to the stream.
- Matched filter: With the assumed distance gradient, creates an approximated absolute magnitude space and applies an isochrone-based matched filter to select members.
>>>>>>> b8f3276d0aed73728d3215506aebd1d8cb25da06

2 Methods: 4D, 6D

4D (x,y,mag,color):
- Use distance fit to filter photometric space
- Use phi2(phi1) function to filter x,y space
- Result: Spatial map with many data points

6D (x,y,mag,color,pmx,pmy):
- Construct membership models and distance fit
- Use distance fit to filter photometric space
- Use phi2(phi1) function to filter x,y space
- Use pmx(phi1) and pmy(phi1) to filter pm space
- Result: Spatial map with sparse data points
