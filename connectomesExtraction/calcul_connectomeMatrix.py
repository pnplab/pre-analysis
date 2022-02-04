#%%

import os
import ast
import pandas as pd
import numpy as np
import math

from nilearn import input_data

import process_connectome as pc
import lib

pd.options.mode.chained_assignment = None

#%%

atlas_name = 'CAB-NP'

atlas_csv_path = os.path.abspath(r'..\..\atlas\CAB-NP_volumetric\atlas_name_labels.csv'.replace('atlas_name',atlas_name))
atlas_csv = pd.read_csv(atlas_csv_path, sep=';', header = 0, names=['Index', 'Region name'], index_col=0)
region_pairwise = int(len(atlas_csv)*(len(atlas_csv)-1)/2)

#%%

directory = 'timeSeries_files'
dataset_name = 'ucla'
sub_dir = os.path.join(directory,dataset_name,atlas_name)

#%%

info_csv_path = os.path.join(sub_dir,'timeSeries_info.csv')
time_series_info = pd.read_csv(info_csv_path, header = 0, index_col=0)
time_series_info = time_series_info.dropna(axis=1)
time_series_info.columns = list(map(lambda x: int(x),time_series_info.columns))
tr = float(time_series_info.loc['tr'].unique()[0])

#%%

task_duration_list = []

for id in time_series_info.columns:
    columns_level = ast.literal_eval(time_series_info.at['nlevels',id])
    time_series_subject = pd.read_csv(time_series_info.at['path', id], header = list(range(columns_level)), index_col=0)
    time_series_subject = time_series_subject.applymap(lambda x: np.array(ast.literal_eval(x)))
    task_duration = lib.calculate_task_durations(time_series_subject) #ast.literal_eval(time_series_info.at['tr', id])
    task_duration_list.append(task_duration)

subject_task_volumes = pd.concat(task_duration_list, axis=1).T
subject_task_volumes.index = time_series_info.columns
subject_task_duration = subject_task_volumes*tr/60

#%%
df = pd.read_csv(r'..\UCLA\subj_tasksDuration.csv', index_col=0)
task_list = df.columns #subject_task_volumes.columns
subject_list = df.index #subject_task_volumes[(subject_task_duration.sum(axis=1)>200)].index
subject_task_volumes = subject_task_volumes.loc[subject_list,task_list].applymap(int)

#%%
delta_t_list = [None]

confounds_name = ['None', 'params9','aCompCor']
confounds_name.remove('None')

directory = 'connectivity_matrix'
sub_dir = os.path.join(directory,dataset_name, atlas_name)
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)

#%%

for delta_t in delta_t_list:

    if delta_t == None:
        subject_task_volumes_min = subject_task_volumes.min(axis=0)
        nb_vol_byTask = pd.DataFrame(np.tile(subject_task_volumes_min.values,(len(subject_task_volumes),1)),
                                     columns=subject_task_volumes.columns, index = subject_task_volumes.index)
        task_proportion_df = nb_vol_byTask.div(nb_vol_byTask.sum(axis=1), axis=0)
        num_matrixConn = 1
    else:
        task_proportion_df = subject_task_volumes.div(subject_task_volumes.sum(axis=1), axis=0)
        time_byTask = task_proportion_df * delta_t
        nb_vol_byTask = time_byTask.applymap(lambda x:math.ceil(x*60/tr))#round(time_byTask * 60/tr).astype(int)
        num_matrixConn = (subject_task_volumes//nb_vol_byTask).astype(int).min().min()

    connectome_matrix = [[np.empty((0, region_pairwise)) for i in range(num_matrixConn)] for j in range(len(confounds_name))]

    for id in subject_task_volumes.index:

        columns_level = ast.literal_eval(time_series_info.at['nlevels',id])
        time_series_subject = pd.read_csv(time_series_info.at['path', id], header = list(range(columns_level)), index_col = 0)
        time_series_subject = time_series_subject.applymap(lambda x: np.array(ast.literal_eval(x)))
        time_series_subject_copy = time_series_subject[task_list].copy()

        for k,index in enumerate(confounds_name):
            time_series_subject_cleaned = time_series_subject_copy.loc[[index]]

            for j in range(num_matrixConn):
                connectome_subject = np.reshape(lib.calculate_conn_subject(time_series_subject_cleaned, task_proportion_df.loc[id,:], nb_vol_byTask.loc[id,:]),(1,region_pairwise))
                connectome_matrix[k][j] = np.vstack((connectome_matrix[k][j],connectome_subject))
                if num_matrixConn is not 1:
                    time_series_subject_cleaned = pc.remove_vol_fromDf(time_series_subject_cleaned,nb_vol_byTask.loc[id,:])

    for k,confound in enumerate(confounds_name):

        conf_dir = os.path.join(sub_dir,confound,str(delta_t))
        if not os.path.exists(conf_dir):
            os.makedirs(conf_dir)

        for j in range(num_matrixConn):
            matrix_csv_path = os.path.join(conf_dir, 'mat_j.csv'.replace('j',str(j)))
            np.savetxt(matrix_csv_path,connectome_matrix[k][j], delimiter = ',')
