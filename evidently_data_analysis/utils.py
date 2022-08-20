from typing import Type
import pandas as pd
from lightning.app.storage.payload import Payload

def check_if_valid_dataframe(df) -> bool:
    if df is not None:
        if isinstance(df, Payload) == False:
            raise TypeError('Passed argument must be of type lightning.app.storage.payload.Payload')
        if isinstance(df.value, pd.DataFrame) == False:
            raise TypeError('The playload must be a valid pandas Dataframe')
    else:
        raise TypeError('Dataframe cannot be NoneType')