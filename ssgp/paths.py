from pathlib import Path


def get_project_path() -> Path:
    return Path(__file__).parent.parent


def get_samples_path() -> Path:
    """ Return path to sample folder """
    return Path(get_project_path(), 'samples')
