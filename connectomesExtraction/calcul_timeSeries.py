import os
import pandas as pd
import dask
from dask.distributed import Client

from nilearn import image
from nilearn import input_data

from bids import BIDSLayout
from load_confounds import Params9
from load_confounds import AnatCompCor

import process_bids as pb
import process_connectome as pc
import lib

pd.options.mode.chained_assignment = None

#%%

atlas_name = 'Yeo_7_WB'
atlas_path = os.path.abspath(r'..\..\atlas\Yeo\Yeo_WB\atlas_name.nii.gz'.replace('atlas_name',atlas_name))

dataset_name = 'msc'
raw_path = os.path.abspath('D:\msc-raw')
derivatives_path = os.path.abspath(r'D:\msc-derivatives\fmriprep')

atlas_masker = input_data.NiftiLabelsMasker(atlas_path, resampling_target="data", standardize=True,
                                            detrend=True, memory='nilearn_cache', memory_level=1)

#%% Load BIDS dataset

layout = BIDSLayout(raw_path, derivatives = derivatives_path, index_metadata=True, reset_database=False)

#%%

subjects_id_list = layout.get_subjects()
tr = layout.get_tr(derivatives=True)

#%%

time_series_info = pd.DataFrame(columns = subjects_id_list, index = ['path','nlevels','tr','atlas'])
time_series_info.loc['tr',:] = tr
time_series_info.loc['atlas',:] = atlas_name

directory = 'timeSeries_files'

sub_dir = os.path.join(directory,dataset_name, atlas_name)
if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)

confounds_name = ['None', 'params9','aCompCor']

#%%

client = Client(threads_per_worker = 4, n_workers = 1)

#%%

results = []
for id in subjects_id_list:
    csv_path = os.path.join(sub_dir,'id_atlas_timeSeries.csv'.replace('id',id).replace('atlas',atlas_name))
    sessions_sub = pb.get_sessions(layout, id)
    tasks_by_session_sub = pb.get_tasks(layout, id, sessions_sub)
    run = pb.get_run(layout, id, sessions_sub, tasks_by_session_sub)
    time_series_df = pc.create_empty_df_timesseries(sessions_sub,tasks_by_session_sub, run, confounds_name)

    # Calcul sur l'ensemble des nifti
    subjects_im_path = layout.get(subject = id, extension='nii.gz', suffix='bold', scope='derivatives',
                                  return_type='filename')

    params9 = Params9().load(subjects_im_path)
    aCompCor = AnatCompCor().load(subjects_im_path)

    time_series_info.loc['path',id] = csv_path
    time_series_info.loc['nlevels',id] = time_series_df.columns.nlevels
    subj_timeseries_df = dask.delayed(lib.calculate_timeseries)(atlas_masker, run, subjects_im_path, csv_path,
                                                            time_series_df, subject_confounds_list = [params9,aCompCor],
                                                            confounds_name= confounds_name)

    results.append(subj_timeseries_df)

#%%

info_csv_path = os.path.join(sub_dir,'timeSeries_info.csv')
time_series_info.to_csv(info_csv_path, header= True)

dask.visualize(results)
dask.compute(*results)
client.close()