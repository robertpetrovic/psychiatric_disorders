# psychiatric_disorders

## Install required packages

``` python -m pip install -r required_packages.txt ```

## How to get your dataset

First, get your disorder of interest label from the clinical.admission.matched.csv columns. For example, if you want to analyze borderline personality disorder, your label would be SCID2PDBPDDX. Input this label when calling FreesurferCluster:

```
disorder = FreesurferCluster('SCID2PDBPDDX')

disorder.matches # returns pairs of IDs of patients and their psychiatric controls
disorder.disorder_pts # returns an ID list of only patients with the disorder of interest
disorder.control_pts # returns an ID list of only psychiatric control patients
disorder.data # returns the volumetric MRI data of the disorder and control patients

```
