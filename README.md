# evidently_data_analysis component

This ⚡ [Lightning component](lightning.ai) ⚡ was generated automatically with:

```bash
lightning init component evidently_data_analysis
```

## What problem is this component solving?
After doing a train-test split of our dataset which is to be used for modelling, this component helps analyse the data trends and any drifts detected in the train and test sets.
In simple words, it gives us a visual analysis to verify if our train and test sets have somewhat similar distributions or not and if they vary then by what degree.

This component uses Evidently AI to facilitate this data drift detection which is one of the crucial steps in the whole ML development lifecycle.

This is made primarily for tabular dataset analysis and currently supported tasks are classification or regression.

This must be done before jumping into modeling and the app can be extended to include the training along with the model prediction analysis too (Psst, Evidently AI helps there as well)

Stop looking at bland numbers which make limited sense, give the visualization a shot.

## To run evidently_data_analysis

First, install evidently_data_analysis (warning: this component has not been officially approved on the lightning gallery):

```bash
lightning install component https://github.com/theUser/evidently_data_analysis
```

Once the app is installed, use it in an app:

```python

import pandas as pd
from evidently_data_analysis import EvidentlyDataAnalysis
from lightning.app.frontend.web import StaticWebFrontend
import lightning as L

class LitApp(L.LightningFlow):
    def __init__(self, train_dataframe_path, test_dataframe_path, target_column_name, task_type) -> None:
        super().__init__()
        self.train_dataframe_path = train_dataframe_path
        self.test_dataframe_path = test_dataframe_path
        self.target_column_name = target_column_name
        self.task_type = task_type
        self.evidently_data_analysis = EvidentlyDataAnalysis(
                                                        train_dataframe_path=self.train_dataframe_path,
                                                        test_dataframe_path=self.test_dataframe_path,
                                                        target_column_name=self.target_column_name,
                                                        task_type=self.task_type)

    def run(self):
        self.evidently_data_analysis.run()

    def configure_layout(self):
        return StaticWebFrontend('./')

if __name__ == "__main__":
    # classification use case
    cancer_df = pd.read_csv('../bcancer.csv')
    total_rows = len(cancer_df)
    train_length = int(total_rows*0.75)

    train_df, test_df = cancer_df[:train_length], cancer_df[train_length:]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_df.to_csv('../ba_cancer_train.csv', index=False)
    test_df.to_csv('../ba_cancer_test.csv', index=False)

    app = L.LightningApp(LitApp(
            train_dataframe_path='../ba_cancer_train.csv',
            test_dataframe_path='../ba_cancer_test.csv',
            target_column_name='target',
            task_type='classification'
        ))

```

## Sample Output of the App

![data_drift](https://user-images.githubusercontent.com/23210132/181892630-7a6afe2f-9ed1-43f3-9425-84c45fb8f665.PNG)

## TODO

[] Write relevant tests
[] Integrate more use cases supported by EvidentlyAI