import os


class Path:
    PROJECT_ROOT = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    LFW_DATA_DIR = os.path.join(DATA_DIR, 'LFW')
    RESOURCES_DIR = os.path.join(PROJECT_ROOT, 'resources')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')
