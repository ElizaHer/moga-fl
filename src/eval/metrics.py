import pandas as pd

class MetricsRecorder:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)

    def to_csv(self, path):
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)
        return df
