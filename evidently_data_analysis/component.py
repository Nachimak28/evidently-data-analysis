from cgi import test
import os
import lightning as L
from lightning.app.storage.payload import Payload
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab, NumTargetDriftTab
import tempfile

class EvidentlyDataAnalysis(L.LightningWork):
    def __init__(self, train_dataframe_path=None, test_dataframe_path=None, target_column_name=None, task_type='classification', parallel=True) -> None:
        super().__init__(parallel=parallel)
        self.train_dataframe_path = train_dataframe_path
        self.test_dataframe_path = test_dataframe_path
        self.target_column_name = target_column_name
        self.task_type = task_type
        self.report_path = None
        tmp_dir = tempfile.mkdtemp()
        self.report_parent_path = os.path.join(tmp_dir, 'data_drift')
        os.makedirs(self.report_parent_path, exist_ok=True)

        self.supported_task_types = ['classification', 'regression']

        if self.task_type not in self.supported_task_types:
            raise Exception(f'task_type must be {",".join(self.supported_task_types)}')


    def run(self, train_df: Payload=None, test_df: Payload=None):
        col_map = ColumnMapping()
        col_map.target = self.target_column_name
        if self.train_dataframe_path == None:
            assert train_df is not None
            train_df = train_df.value
        else:
            train_df = pd.read_csv(self.train_dataframe_path)
        
        if self.test_dataframe_path == None:
            assert test_df is not None
            test_df = test_df.value
        else:
            test_df = pd.read_csv(self.test_dataframe_path)


        tabs = [DataDriftTab(verbose_level=0)]
        if self.task_type == 'classification':
            tabs.append(CatTargetDriftTab(verbose_level=0))
        else:
            tabs.append(NumTargetDriftTab(verbose_level=0))
        data_and_target_drift_dashboard = Dashboard(tabs=tabs)
        data_and_target_drift_dashboard.calculate(train_df, test_df, column_mapping=col_map)
        self.report_path = os.path.join(self.report_parent_path, 'index.html')
        data_and_target_drift_dashboard.save(self.report_path)
