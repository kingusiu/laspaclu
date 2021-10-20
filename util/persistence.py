import os
import datetime
import pathlib


def make_model_path(date=None, prefix='AE', run_n=0, mkdir=False):
    date_str = ''
    if date is None:
        date = datetime.date.today()
        date = '{}{:02d}{:02d}'.format(date.year, date.month, date.day)
    path = os.path.join('models/saved', prefix+'model_run{}_{}'.format(str(run_n), date))
    if mkdir:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path
