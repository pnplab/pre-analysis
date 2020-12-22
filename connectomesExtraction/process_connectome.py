import pandas as pd
import numpy as np

from nilearn import connectome
from nilearn import image


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
