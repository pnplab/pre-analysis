import numpy as np
import itertools
from bids.layout import parse_file_entities


def get_sessions(layout, subject_id):
    # get sessions list from one subject ID
    session_list = layout.get(subject=subject_id, return_type='id', target='session',
                              suffix='bold', scope='derivatives')
    return np.array(session_list)


def get_tasks(layout, subject_id, session_list):
    # get tasks list from one subject ID and session ID

    if session_list.size == 0:
        tasks_list = layout.get(subject=subject_id, return_type='id', target='task',
                                suffix='bold', scope='derivatives')

    else:
        tasks_list = []
        for session in session_list:
            tasks_list_sess = layout.get(subject=subject_id, session=session, return_type='id', target='task',
                                         scope='derivatives', suffix='bold')
            tasks_list.append(tasks_list_sess)
    return np.array(tasks_list)


def get_run(layout, subject_id, session_list, task_list):
    run_list = []

    if session_list.size == 0:
        for task in task_list:
            run_list_task = layout.get(subject=subject_id, task=task, return_type='id', target='run',
                                       suffix='bold', scope='derivatives')
            run_list.append(run_list_task)

        if all([not elem for elem in run_list]):
            run_list = []
        elif any([not elem for elem in run_list]):
            run_list = [[0] if x == [] else x for x in run_list]

    else:
        for i in range(len(session_list)):
            run_list_session = []
            task_list_session = task_list[i]
            for task in task_list_session:
                run_list_task = layout.get(subject=subject_id, session=session_list[i], task=task,
                                           return_type='id', target='run', suffix='bold', scope='derivatives')
                run_list_session.append(run_list_task)
            run_list.append(run_list_session)

        run_list_flatten = list(itertools.chain.from_iterable(run_list))
        if all([not elem for elem in run_list_flatten]):
            run_list = []
        elif any([not elem for elem in run_list_flatten]):
            for i in range(len(session_list)):
                task_list_session = task_list[i]
                for j in range(len(task_list_session)):
                    if not run_list[i][j]:
                        run_list[i][j] = [0]
    return np.array(run_list)


def get_entities(image_path, run_by_task_sub):
    image_caract = parse_file_entities(image_path)
    if 'run' not in image_caract.keys() and run_by_task_sub.size != 0:
        image_caract['run'] = 0

    return image_caract


def get_keys_of_interest(image_caract):
    keys_of_interest = [image_caract.get('task'), image_caract.get('session'), image_caract.get('run')]

    if None in keys_of_interest:
        keys_of_interest = list(filter(None, keys_of_interest))

    return tuple(keys_of_interest)



