# The Python Scripts and Jupyter Notebooks used for my Master's Thesis

The validation tests file contains the jupyter notebooks used for the validation tests and methodology section of my thesis. 
* [The NYU-VAGC file can be found here](https://sdss.physics.nyu.edu/vagc/), specifically the .csv file using nearest petromags corrected to 00.
* [The cross-footprint file can be retrieved from here](https://datalab.noirlab.edu/data-explorer), using the SQL Query command included in the Appendix of my thesis.

*'schechter.py'* is the python script that has been used to compute the Schechter parameters for each redshift bin across our range. To use this script, *'sch.sh'* must be executed as it defines the requirements for each job and the job ID is passed to *'schechter.py'* using *argparse* to define the edges of the redshift bin for that job. Both the script and shell command file are based on **Slurm** system for the UZH S3IT Science Cluster.

*'DES_Y3_datavis.ipynb'* is the Jupyter notebook used to visualize the sky regions we create using *[kmeansradec](https://github.com/esheldon/kmeans_radec?tab=readme-ov-file)* and the color-magnitude distribution of the galaxies in our sample.

*'completeness.ipynb'* is the Jupyter notebook used to test our sample and compare our results to other ones from the literature. By far, it is the most confusing file in this repository as it contains many method tests and visualizations in each step to ensure the correct execution of our methods. It consists of two main sections, estimation of the Schechter function and different completeness computations based on the methods described in the thesis.

I will not be providing the DES Y3 Gold catalog here as the file is too large, however it can be downloaded from [CosmoHub](https://cosmohub.pic.es/login) as a '.fits' file, which you can tranform to a '.hdf5' file or adapt the data loading for it in the code. For my results, you can find the Schechter parameters, their standard deviations and the fitting ranges for each of them as '.txt' files, under the aptly named 'Results' file. '*02_22*' refers to the redshift range (0.02,0.22], and the numbered ones are based on the job ID used in *'sch.sh'*, referring to the five distinct bin ranges. Each of them have headers that explain their composition in more detail.
