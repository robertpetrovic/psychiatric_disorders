from jake import *
import pickle

import pandas as pd
import numpy as np
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind

class FreesurferCluster():
    def __init__(self, disorder:str, normalize=True):
        '''
        Using the clinical comorbidity dataset, extracts IDs of patients with a 
        disorder and IDs of psychiatric controls. The respective Freesurfer data
        of these patients is then returned.
        
        Parameters
        ----------
        disorder: string
            The disorder of interest. Must be one of the column labels of the
            clinical comorbidities dataset.
        normalize: bool (default: True)
            Whether or not the returned dataframe will be normalized.
        
        Returns
        -------
        data: pd.DataFrame
            Dataframe of Freesurfer data from patients and psychiatric controls,
            appended a 'class' column (0 denotes control, 1 denotes disorder of interest)
        '''
        self.disorder = disorder
        self.matcher = PatientMatcher(excel_filepath='clinical.admission.matched.csv', disorder=disorder)
        self.matches = self.matcher.get_best_matches()
        self.disorder_pts = self.matches.control_id.index.tolist()
        self.control_pts = self.matches.control_id.values.tolist()
        self.all_pts = list(set(self.disorder_pts) | set(self.control_pts))
        
        df = pd.read_csv('Freesurfer_V1.matched.csv', encoding = "ISO-8859-1", low_memory=False)
        df = df.set_index('SubjectID')
        df = df.loc[:,df.dtypes.values!='object']
        df = df[df.notna().any(axis=1)] # takes patients with at least some data
        df = df.loc[:,df.notna().all()] # filters out columns that have no data
        
        df = df.loc[:,[len(np.unique(df[col]))>1 for col in df.columns]] # essentially filters out st dev of 0
           
        self.mean = df.mean()
        self.std = df.std()
        
        df = df.loc[list(set(df.index) & set(self.all_pts)), :]
        
        self.class_labels = [int(i) for i in df.index.isin(self.disorder_pts)]
        df['class'] = self.class_labels
        self.data = df

        
    def data_transforms(self, mean_std=False, eTIV=False, t_test=False):
        '''
        Allows for different normalization techniques for data exploration purposes.
        
        Parameters
        ----------
        mean_std: bool (default: False)
            Whether or not to normalize data column-wise by subtracting the mean then dividing 
            by the standard deviation.
        eTIV: bool (default: False)
            Whether or not to normalize data row-wise by dividing with each patient's eTIV.
        t_test: bool (default: False)
            Whether or not to filter only features with a significant difference between DOI
            patients and psychiatric controls. p-value threshold used is 0.05
        '''
        df = self.data
        
        if mean_std:
            df = df.sub(self.mean).div(self.std).loc[:,df.columns]
            
        if eTIV:
            dont_normalize = ['meancurv','std','eTIV','class']
            df_norm = df.loc[:,~df.columns.str.contains('|'.join(dont_normalize))].div(df.eTIV,axis=0)
            df_no_norm = df.loc[:,df.columns.str.contains('|'.join(dont_normalize))]
            df = pd.concat([df_norm,df_no_norm],axis=1)
            
        if t_test:
            control_df = df.loc[list(set(df.index) & set(self.control_pts)),:]
            disorder_df = df.loc[list(set(df.index) & set(self.disorder_pts)),:]
            t_tests = [ttest_ind(control_df[col],disorder_df[col]).pvalue for col in df.columns.tolist()]
            t_tests = pd.Series(t_tests)
            t_tests.index = df.columns.tolist()
            sig_vars = t_tests[t_tests<0.05].index
            df = df[sig_vars]
        
        df.loc[:,'class'] = self.class_labels
        return df
        
    def cluster(self, algo_method = 'PCA'):
        '''
        A first-pass to see if the Freesurfer data can be clustered by conventional algorithms.
        
        Parameters
        ----------
        algo_method: string (default: PCA)
            The type of algorithm used to visualize the patient data in 2 dimensions. Must be
                - 'PCA' (principal component analysis)
                - 'MDS' (multi-dimensional scaling)
                - 'TSNE' (t-distributed stochastic neighbor embedding)
        Returns
        -------
        Plot of Freesurfer data in 2 dimensions.
        '''
        if algo_method == 'PCA':
            algo = PCA
        elif algo_method == 'MDS':
            algo = MDS
        elif algo_method == 'TSNE':
            algo = TSNE
        else:
            raise ValueError(
                f'Unrecognized decomposition method "{algo_method}"')
        
        algo = algo(n_components=2,random_state=42)
        
        decomp = pd.DataFrame(
            algo.fit_transform(self.data),
            index=self.data.index,
            columns=['dim1', 'dim2'])

        ax = plt.subplots(1, 1, figsize=(6, 6))[1]
        decomp[decomp.index.isin(self.control_pts)].plot.scatter(
            'dim1', 'dim2', c='C0', ax=ax, label='control')
        if self.disorder[-3:]=='CUR':
            label = self.disorder[:-3].lower()
        else:
            label = self.disorder[:-2].lower()
        decomp[decomp.index.isin(self.disorder_pts)].plot.scatter(
            'dim1', 'dim2', c='C1', ax=ax, label=label)
        plt.show()
        