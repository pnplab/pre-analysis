import numpy as np
import process_connectome as pc
import process_bids as pb

from nilearn import image
from nilearn import signal


# %%
def calculate_timeseries(atlas_masker, run_by_task_sub, subjects_im_path, csv_path, time_series_df,
                         subject_confounds_list=None, index_name=None):
    for i, image_path in enumerate(subjects_im_path):
        image_caract = pb.get_entities(image_path, run_by_task_sub)
        image_koi = pb.get_keys_of_interest(image_caract)

        im = image.load_img(image_path)
        time_series = atlas_masker.fit_transform(im, confounds=None)
        time_series_df.loc['None', image_koi] = time_series.tolist()

        for j, confounds in enumerate(subject_confounds_list):
            time_series_cleaned = signal.clean(time_series, confounds=confounds[i])
            time_series_df.loc[index_name[j + 1], image_koi] = time_series_cleaned.tolist()

    time_series_df = time_series_df.replace(r'^\s*$', np.nan, regex=True)  # to delete
    if time_series_df.isnull().any().any():
        time_series_df = time_series_df.dropna(axis='columns')

    time_series_df.to_csv(csv_path, header=True)

    return None


# %%

def calculate_task_durations(time_series_df):

    temp = time_series_df.applymap(len)
    time_series_caract = temp.drop_duplicates(ignore_index=True)
    task_duration = []

    if time_series_caract.shape[0] == 1:

        time_series_caract.loc['duration'] = time_series_caract.loc[0]

        if time_series_df.columns.nlevels == 1:
            task_duration = time_series_caract.loc['duration']
        else:
            task_duration = time_series_caract.loc['duration'].groupby(level=0).sum()

    else:
        print('Length error: error in time series calculation')

    return task_duration


# %%

def calculate_conn_subject(time_series_df, task_proportion, nb_vol_byTask):
    conn_list = []

    for task_name in task_proportion.index:
        time_series_df_task = time_series_df.loc[:, task_name]

        time_series_connectome, nb_vol_connectome = pc.get_volumes(nb_vol_byTask[task_name], time_series_df_task)

        if 1 in nb_vol_connectome:
            pos = np.where(np.array(nb_vol_connectome) == 1)[0]
            nb_vol_connectome = np.delete(np.array(nb_vol_connectome), pos)
            time_series_connectome = np.delete(np.array(time_series_connectome), pos).tolist()

        weights = np.divide(nb_vol_connectome, nb_vol_byTask[task_name])

        conn = pc.mean_weighted_connectome(time_series_connectome, weights)
        conn_list.append(conn)

    conn_subj = np.average(conn_list, axis=0, weights=task_proportion.values)

    return conn_subj
