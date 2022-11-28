# psychiatric_disorders

## Install required packages

```berry
python -m pip install -r required_packages.txt
```

## How to access your data

First, get your disorder of interest label from the clinical.admission.matched.csv columns. For example, if you want to analyze borderline personality disorder, your label would be SCID2PDBPDDX. Input this label when calling FreesurferCluster:

```berry
disorder = FreesurferCluster('SCID2PDBPDDX')
```

The result contains many attributes useful for data analysis:

```berry
disorder.matches # returns pairs of IDs of patients and their psychiatric controls
disorder.disorder_pts # returns an ID list of only patients with the disorder of interest
disorder.control_pts # returns an ID list of only psychiatric control patients
disorder.data # returns the volumetric MRI data of the disorder and control patients
```
The result also has methods (functions):
```berry
disorder.data_transforms(mean_std = True,
                         eTIV = True,
                         t_test = True) # returns the volumetric MRI data after normalizing and filtering features
disorder.cluster(algo_method = 'PCA') # visualization of MRI data clustering via an algorithm of choice
```
* Note: the data_transforms() method does not overwrite the original dataframe.
