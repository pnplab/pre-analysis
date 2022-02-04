import os
import ast
import time
import pandas as pd
import numpy as np

from nilearn import image
from nilearn import input_data
from nilearn import signal

from bids import BIDSLayout
from load_confounds import Params9
from load_confounds import AnatCompCor

import lib

pd.options.mode.chained_assignment = None

# %% Importation des données

msc_raw_path = os.path.abspath('D:\msc-raw')
msc_derivatives_path = os.path.abspath(r'D:\msc-derivatives\fmriprep')

# Lecture du format BIDS
layout = BIDSLayout(msc_raw_path, derivatives=msc_derivatives_path, validate=True,
                    database_path='pybids_True.sql', index_metadata=True, reset_database=False)
print('BIDS layout ok')

# %% Caractéristique du dataset

subjects_id_list = layout.get_subjects()
tr = layout.get_tr(derivatives=True)
time_series_info = pd.DataFrame(columns=subjects_id_list, index=['path','nlevels','tr','atlas'])
time_series_info.loc['tr',:] = tr

# %% Atlas choisi pour la parcellisation

atlas_name = 'CAB-NP'
time_series_info.loc['atlas',:] = atlas_name
atlas_path = os.path.abspath( r'C:\Users\smine\OneDrive\Documents\GitHub\atlas\CAB-NP_volumetric\CAB-NP_volumetric_liberal.nii.gz')
atlas_masker = input_data.NiftiLabelsMasker(atlas_path, resampling_target="data", standardize=True,
                                            detrend=True, memory='nilearn_cache', memory_level=1)

atlas_csv_path = os.path.abspath(r'C:\Users\smine\OneDrive\Documents\GitHub\atlas\CAB-NP_volumetric\CAB-NP_labels.csv')
atlas_csv = pd.read_csv(atlas_csv_path, sep=';')
connectome_length = int(len(atlas_csv)*(len(atlas_csv)-1)/2)

# %% analyse sujet 1

subjects_im_path = layout.get(subject=subjects_id_list[8], extension='nii.gz', suffix='bold', scope='derivatives',
                                return_type='filename')

sessions_sub = lib.get_sessions(layout, subjects_id_list[9])
tasks_by_session_sub = lib.get_tasks(layout, subjects_id_list[9], sessions_sub)
run_by_task_sub = lib.get_run(layout, subjects_id_list[9], sessions_sub, tasks_by_session_sub)

#%% tsv

params9 = Params9().load(subjects_im_path)
aCompCor = AnatCompCor().load(subjects_im_path)
index_name = ['None', 'params9','aCompCor']

# %% Creation du dataframe vide

time_series_df = lib.create_empty_df_timesseries(sessions_sub, tasks_by_session_sub, run_by_task_sub, index_name)

# %% Conversion d'un nifti en time series

### bids format processing
# image_path = subjects_im_path[0]
# image_caract = lib.get_entities(image_path,run_by_task_sub)
# image_koi = lib.get_keys_of_interest(image_caract)
#
# time_series = lib.apply_masker(atlas_masker, image_path)
#
# time_series_cleaned_param9 = signal.clean(time_series, confounds=params9[0])
# time_series_cleaned_acc = signal.clean(time_series, confounds=aCompCor[0])
#
# time_series_df.loc['None', image_koi] = time_series
# time_series_df.loc['params9', image_koi] = time_series_cleaned_param9
# time_series_df.loc['aCompCor', image_koi] = time_series_cleaned_acc


# %% Calcul sur l'ensemble des nifti du sujet 1

for i, image_path in enumerate(subjects_im_path):
    image_caract = lib.get_entities(image_path, run_by_task_sub)
    image_koi = lib.get_keys_of_interest(image_caract)

    time_series = lib.apply_masker(atlas_masker, image_path)

    time_series_cleaned_param9 = signal.clean(time_series, confounds=params9[i])
    time_series_cleaned_acc = signal.clean(time_series, confounds=aCompCor[i])

    time_series_df.loc['None', image_koi] = time_series.tolist()
    time_series_df.loc['params9', image_koi] = time_series_cleaned_param9.tolist()
    time_series_df.loc['aCompCor', image_koi] = time_series_cleaned_acc.tolist()

if time_series_df.isnull().any().any():
    time_series_df = time_series_df.dropna(axis='columns')

#%%
time_series_info.loc['nlevels',subjects_id_list[9]] = time_series_df.columns.nlevels

csv_path = os.path.join('timeSeries_files', 'MSC06_CAB-NP_timeSeries.csv')
time_series_info.loc['path',subjects_id_list[9]] = csv_path
time_series_df.to_csv(csv_path, header=True)

#%%
time_series_df = pd.read_csv(csv_path, header=list(range(3)), index_col=0)
time_series_df.columns = time_series_df.columns.set_levels(time_series_df.columns.levels[2].astype(int), level=2)
time_series_df = time_series_df.applymap(lambda x: np.array(ast.literal_eval(x)))

#%% Time series information

task_duration = lib.calculate_task_durations(time_series_df, tr)

task_list = list(task_duration.index)
task_proportion = task_duration[task_list].div(task_duration[task_list].sum())

delta_t = 60  # minutes
delta_t_byTask = task_proportion * delta_t
nb_vol_byTask = round(delta_t_byTask * 60 / tr).astype(int)  # seconds

# %%
num_matrixConn = round((task_duration // delta_t_byTask).min())
connectome_matrix = [np.empty((0, connectome_length)) for i in range(num_matrixConn)]

# %%
time_series_df_copy = time_series_df[task_list].copy()

for j in range(num_matrixConn):
    connectome_subject = np.reshape(lib.calculate_conn_subject(time_series_df_copy, task_proportion, nb_vol_byTask),
                                    (1,connectome_length))
    connectome_matrix[j] = np.vstack((connectome_matrix[j], connectome_subject))
    time_series_df_copy = lib.remove_vol_fromDf(time_series_df_copy, nb_vol_byTask)
    print(time_series_df_copy.shape[1])
