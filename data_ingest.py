import pandas as pd

class IngestData:
    def __init__(self) -> None:
        self.data_path = None
        
    def get_data(self, data_path: str) -> pd.DataFrame:
        self.data_path = data_path
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding='latin-1')
            
        print("Available columns:", df.columns.tolist())
        return df