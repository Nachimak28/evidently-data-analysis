from lightning.app.storage.payload import Payload
import pandas as pd
from evidently_data_analysis import EvidentlyDataAnalysis
from lightning.app.frontend.web import StaticWebFrontend
import lightning as L

class LoadData(L.LightningWork):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_df = None
        self.test_df = None

    def run(self):
        self.train_df = Payload(pd.read_csv('resources/ba_cancer_train.csv'))
        self.test_df = Payload(pd.read_csv('resources/ba_cancer_test.csv'))

class LitApp(L.LightningFlow):
    def __init__(self, train_dataframe_path=None, test_dataframe_path=None, target_column_name=None, task_type=None) -> None:
        super().__init__()
        self.train_dataframe_path = train_dataframe_path
        self.test_dataframe_path = test_dataframe_path
        self.target_column_name = target_column_name
        self.task_type = task_type
				# Flask won't return 
        self.evidently_data_analysis = EvidentlyDataAnalysis(parallel=True, cloud_compute=L.CloudCompute("default"))
        self.load_data_component = LoadData(parallel=True, cloud_compute=L.CloudCompute("default"))

    def run(self):
        self.load_data_component.run()
        self.evidently_data_analysis.task_type = 'classification'
        self.evidently_data_analysis.target_column_name = 'target'
				# wait for load_data_component make Payload available
        if self.load_data_component.train_df and self.load_data_component.test_df:
          self.evidently_data_analysis.run(train_df=self.load_data_component.train_df, test_df=self.load_data_component.test_df)

    def configure_layout(self):
			# wait for work to complete so that manual refresh is not required
      if self.evidently_data_analysis.report_path:
        tab_1 = {'name': 'Data report', 'content': self.evidently_data_analysis}
      else:
        tab_1 = []
      return tab_1

if __name__ == "__main__":
    # classification use case
    # app = L.LightningApp(LitApp(
    #         train_dataframe_path='resources/ba_cancer_train.csv',
    #         test_dataframe_path='resources/ba_cancer_test.csv',
    #         target_column_name='target',
    #         task_type='classification'
    #     ))

    app = L.LightningApp(LitApp())

    # regression use case

    # app = L.LightningApp(LitApp(
    #         train_dataframe_path='resources/ca_housing_train.csv',
    #         test_dataframe_path='resources/ca_housing_test.csv',
    #         target_column_name='MedHouseVal',
    #         task_type='regression'
    #     ))

