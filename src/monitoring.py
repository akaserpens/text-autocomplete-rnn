from clearml import Task
import os

CLEARML_ENABLED = False

def init_clearml(config):
    CLEARML_ENABLED = bool(config['enabled'])
    if not CLEARML_ENABLED:
        return
    Task.set_credentials(
        api_host=config['api_host'],
        web_host=config['web_host'],
        files_host=config['files_host'],
        key=os.getenv('CLEARML_KEY'),
        secret=os.getenv('CLEARML_SECRET')
    )

class ClearMLTask:
    def __init__(self, project_name: str, task_name: str, task_config: dict):
        self.task_config: dict = task_config
        self.project_name: str = project_name
        self.task_name: str = task_name
        self.task: Task = None

    def __enter__(self):
        if CLEARML_ENABLED:
            self.task = Task.init(project_name=self.project_name, task_name=self.task_name)
            self.task.connect(self.task_config)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.task is not None:
            self.task.close()

    def report_scalar(self, title: str, series: str, value: float, iteration: int):
        if self.task is not None:
            self.task.get_logger().report_scalar(title=title, series=series, value=value, iteration=iteration)