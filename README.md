# evidently_data_analysis component

This ⚡ [Lightning component](lightning.ai) ⚡ was generated automatically with:

```bash
lightning init component evidently_data_analysis
```

## To run evidently_data_analysis

First, install evidently_data_analysis (warning: this component has not been officially approved on the lightning gallery):

```bash
lightning install component https://github.com/theUser/evidently_data_analysis
```

Once the app is installed, use it in an app:

```python
from evidently_data_analysis import TemplateComponent
import lightning as L


class LitApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.evidently_data_analysis = TemplateComponent()

    def run(self):
        print("this is a simple Lightning app to verify your component is working as expected")
        self.evidently_data_analysis.run()


app = L.LightningApp(LitApp())
```
