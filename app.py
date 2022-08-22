from lightning.app.storage.payload import Payload
import pandas as pd
from evidently_data_analysis import EvidentlyDataAnalysis
from lightning.app.frontend.web import StaticWebFrontend
import lightning as L

class StaticPageViewer(L.LightningFlow):
    def __init__(self, page_path: str):
        super().__init__()
        self.serve_dir = page_path

    def configure_layout(self):
        return StaticWebFrontend(serve_dir=self.serve_dir)


class TempWorkComponent(L.LightningWork):
    def __init__(self, parallel=True) -> None:
        super().__init__(parallel=parallel)
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
        self.evidently_data_analysis = EvidentlyDataAnalysis()



        self.report_render = StaticPageViewer(self.evidently_data_analysis.report_parent_path)
        self.temp_component = TempWorkComponent(parallel=False)

    def run(self):
        self.temp_component.run()
        self.evidently_data_analysis.task_type = 'classification'
        self.evidently_data_analysis.target_column_name = 'target'

        self.evidently_data_analysis.run(train_df=self.temp_component.train_df, test_df=self.temp_component.test_df)

    def configure_layout(self):
        tab_1 = {'name': 'Data report', 'content': self.report_render}
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

