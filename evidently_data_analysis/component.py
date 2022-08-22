import os
import logging
import tempfile
import pandas as pd
from typing import Optional

import lightning as L
from lightning.app.storage.payload import Payload

from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab, NumTargetDriftTab

from .utils import check_if_valid_dataframe

class EvidentlyDataAnalysis(L.LightningWork):
    def __init__(self, 
                train_dataframe_path: Optional[str]=None, 
                test_dataframe_path: Optional[str]=None, 
                target_column_name: Optional[str]=None, 
                task_type: str='classification', 
                parallel: bool=False, 
                report_parent_path: str=None) -> None:
        """
        A Simple Lightning component which uses EvidentlyAI to generate an interactive data dashboard for train and test data.
        The data report is stored in an HTML file to the path of your choosing or a temp path
        It is build for tabular datasets to analyse model drift in the train data and incoming test/production data.
        This component works in 2 ways - passing arguments during initialization or before running the run method.
        Note: If arguments are passed during the initialization, call the run method without any args.
        The generated report can be found at the report_path attribute of the object of this class.

        Parameters
        ----------
            train_dataframe_path : str (optional)
                Path to train dataframe (CSV file)
            test_dataframe_path : str (optional)
                Path to test dataframe (CSV file)
            target_column_name : str (optional)
                Name of the target/label column in the train and test dataframes - must be the same for both
            task_type : str
                Type of task - classification/regression (only these 2 supported as of now)
            parallel : boolean
                Flag to indicate if this component is to be run in parallel or not with other components - useful when integrating with other components
            report_parent_path : str (optional)
                Provide a directory to write the data report. If None, then a temp directory is used

        Returns
        -------
        None
        """
        super().__init__(parallel=parallel)

        self.train_dataframe_path = train_dataframe_path
        self.test_dataframe_path = test_dataframe_path
        self.target_column_name = target_column_name
        self.task_type = task_type
        self.report_path = None                             # the path of the generated report HTML file
        
        # report path setup
        if report_parent_path:
            self.report_parent_path = report_parent_path
        else:
            tmp_dir = tempfile.mkdtemp()
            self.report_parent_path = os.path.join(tmp_dir, 'data_drift')
            os.makedirs(self.report_parent_path, exist_ok=True)

        # supported task types
        self.supported_task_types = ['classification', 'regression']

        # checking if provided task type is among the supported task types
        if self.task_type not in self.supported_task_types:
            raise Exception(f'task_type must be {",".join(self.supported_task_types)}')


    def run(self, train_df: Optional[Payload]=None, test_df: Optional[Payload]=None) -> None:
        """
        Builds the EvidentlyAI data dashboard. The run method allows passing of the data arguments 
        because if in some case, the data is not available beforehand (eg: uploading CSV files from a Gradio component) then
        they can be passed during the run method call.

        Parameters
        ----------
        train_df : lightning.storage.payload.Payload
            The training dataframe passed as a pandas dataframe object
        train_df : lightning.storage.payload.Payload
            The testing dataframe passed as a pandas dataframe object
        
        """
        # build the report path dir if it does not exist already - sanity check
        os.makedirs(self.report_parent_path, exist_ok=True)

        # define the column mapping to their type (categorical/numeric/target etc)
        # evidently generally discovers the column types by itself but passing can be set explicitly
        col_map = ColumnMapping()
        col_map.target = self.target_column_name

        # check if the train and test dataframes are passed during init or run and use accordingly
        if self.train_dataframe_path == None:
            check_if_valid_dataframe(train_df)
            train_df = train_df.value
        else:
            if os.path.exists(self.train_dataframe_path):
                train_df = pd.read_csv(self.train_dataframe_path)
            else:
                raise FileNotFoundError('Train dataframe path does not exist, please provide valid path')
        
        if self.test_dataframe_path == None:
            check_if_valid_dataframe(test_df)
            test_df = test_df.value
        else:
            if os.path.exists(self.test_dataframe_path):
                test_df = pd.read_csv(self.test_dataframe_path)
            else:
                raise FileNotFoundError('Test dataframe path does not exist, please provide valid path')

        # verify if target column present in both train and test dfs
        assert self.target_column_name in train_df.columns
        assert self.target_column_name in test_df.columns

        # assert if no nan/null values are present
        assert train_df.isnull().values.any() == False, 'Nan/Null values not allowed'
        assert test_df.isnull().values.any() == False, 'Nan/Null values not allowed'

        logging.info('Dataframe validation successful')

        # tabs to be included in the dashboard 
        tabs = [DataDriftTab(verbose_level=0)]

        if self.task_type == 'classification':
            tabs.append(CatTargetDriftTab(verbose_level=0))
        else:
            tabs.append(NumTargetDriftTab(verbose_level=0))
        
        data_and_target_drift_dashboard = Dashboard(tabs=tabs)
        data_and_target_drift_dashboard.calculate(train_df, test_df, column_mapping=col_map)

        # the HTML file must be names index.html only for lightning component to pick and render it
        # Reference: https://lightning.ai/lightning-docs/api_reference/generated/lightning_app.frontend.web.StaticWebFrontend.html#lightning_app.frontend.web.StaticWebFrontend
        self.report_path = os.path.join(self.report_parent_path, 'index.html')
        
        # save the report to the path
        data_and_target_drift_dashboard.save(self.report_path)
        logging.info('Dashboard generated successfully')
