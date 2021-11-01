import os
import ast
import pandas as pd
import numpy as np
import process_bids as pb

from nilearn import image
from nilearn import signal
from nilearn import connectome

#%%
def create_empty_df_timesseries(sessions_sub, tasks_by_session_sub, run_by_task_sub, index):
    list_tuple = []

    if sessions_sub.size == 0 and run_by_task_sub.size == 0:
        column = tasks_by_session_sub

    elif sessions_sub.size == 0:
        for j in range(len(tasks_by_session_sub)):
            task = tasks_by_session_sub[j]
            run_task = run_by_task_sub[j]
            for k in range(len(run_task)):
                run_num = run_task[k]
                list_tuple.append(tuple([task, run_num]))

        column = pd.MultiIndex.from_tuples(list_tuple)

    else:
        for i in range(len(sessions_sub)):
            session = sessions_sub[i]
            task_list = tasks_by_session_sub[i]
            for j in range(len(task_list)):
                task = task_list[j]

                if run_by_task_sub.size == 0:
                    list_tuple.append(tuple([session, task]))
                else:
                    run_task = run_by_task_sub[i][j]
                    for k in range(len(run_task)):
                        run_num = run_task[k]
                        list_tuple.append(tuple([session, task, run_num]))

        column = pd.MultiIndex.from_tuples(list_tuple)
        column = column.swaplevel(0,1).sortlevel(0)[0]

    time_series_df = pd.DataFrame(columns=column, index=index)
    return time_series_df

#%%

def calculate_timeseries(atlas_masker, run_by_task_sub, subjects_im_path, csv_path, time_series_df, t_r,
                         subject_confounds_list = None, confounds_name = None):

    for i, image_path in enumerate(subjects_im_path):
        image_caract = pb.get_entities(image_path, run_by_task_sub)
        image_koi = pb.get_keys_of_interest(image_caract)

        im = image.load_img(image_path)
        time_series = atlas_masker.fit_transform(im, confounds=None)
        time_series_df.loc['None',image_koi] = time_series.tolist()

        for j, confounds in enumerate(subject_confounds_list):
            time_series_cleaned = signal.clean(time_series, confounds=confounds[i],
                                               detrend=False, standardize='zscore', t_r=t_r)
            time_series_df.loc[confounds_name[j+1], image_koi] = time_series_cleaned.tolist()

    time_series_df = time_series_df.replace(r'^\s*$', np.nan, regex=True) # to delete
    if time_series_df.isnull().any().any():
        time_series_df = time_series_df.dropna(axis='columns')

    time_series_df.to_csv(csv_path, header=True)
    return None

#%%

def extract_connectomes(time_series_info, directory, confounds_name, kind='correlation', vectorize= True,
                        discard_diagonal=True):
    for id in time_series_info.columns:
        columns_level = ast.literal_eval(time_series_info.at['nlevels', id])
        time_series_subject = pd.read_csv(time_series_info.at['path', id], header=list(range(columns_level)),
                                          index_col=0)
        time_series_subject = time_series_subject.applymap(lambda x: np.array(ast.literal_eval(x)))

        for caract in time_series_subject.columns:
            time_series_subject_caract = time_series_subject.loc[confounds_name, caract]
            correlation_measure = connectome.ConnectivityMeasure(kind=kind, vectorize=vectorize,
                                                                 discard_diagonal=discard_diagonal)
            connectomes_matrix = correlation_measure.fit_transform(time_series_subject_caract)

            for i, confound in enumerate(confounds_name):
                connectome_folder_path = os.path.join(directory, confound)
                connectome_csv_path = os.path.join(connectome_folder_path, '_'.join((id,) + caract) + '.csv')
                pd.DataFrame(connectomes_matrix[i]).to_csv(connectome_csv_path, index=False)
    return
