
Application for ADCP Data Analysis  
=======================================

About
=====
An Acoustic Doppler Current Profiler (ADCP) is sound wave based current velocity measuring instrument. For more information, see [ADCP](https://www.whoi.edu/what-we-do/explore/instruments/instruments-sensors-samplers/acoustic-doppler-current-profiler-adcp/).  

Demo
====

![Demo file](./png/demo.gif)
 
TODO:
=====
- [x]  Allow multiple deployments to be selected for comparison.

- [x]  Consider breaking down data into seasons/months for proper comparison of deployments. For example Jul-Sept 
flow could be quite different from the winter flow. Allow users to select seasons/months etc.

- [ ] Add information on when ( at deployment or recovery) fixed orientation information was obtained

- [ ] Use  PCA ellipses as the main tools going forward for decision making on the need to correct orientations.  

- [ ] Keep in mind the  deep water renewal events in the SoG central that are in phase with the neap tides when 
calculating any statistic for directions or speed. 

- [ ] Collect deployment and principal flow related data for all adcp sites from Richard D. 

- [ ] Add the best bathymetry (multi-beam) contour map with the exact/best estimate of where each deployment was
and what the local “heading” is of the isobaths. While this is only a reassuring piece of information, 
we cannot have the currents near the bottom heading into or out of the slope, they would align themselves to 
be very close to parallel with the isobaths. So another column in the table would be the isobath heading at 
the deployment site. 

- [ ] Add PCA ellipses on a map with centres at the site lon,lat.

- [ ] Build a database for long term stats so we can easily compare deployments.

- [ ] Store data where easily accessible and cheap to do so (aws S3 bucket is good for storage but not sure about the cost since they charge by number of transfers out of the bucket).

- [ ] Think about faster ways to calculate everything, cache everything when possible and appropriate.

- [ ] Allow users to drag and align ellipses on the map, and the save new parameters. That would be a nice feature.

- [ ] Add scripts to download data using the api. The scripts should be able to download data from the server and save it to 
the local disk in the appropriate directory. If data file is to be manually updated, the directory structure should look like the one below 

 
 

 

