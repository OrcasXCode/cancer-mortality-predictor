#andas is used for data manipulation and analysis, especially with tabular data (like CSV files).
import pandas as pd

class IngestData:
    def __init__(self) -> None:
        self.data_path = None
        
    #Python automatically passes ingester (the object itself) as the first argument to the method — that's what self represents.
    #If you remove self, the method wouldn’t know which instance it’s working on, and you couldn’t access or modify instance variables like data_path.
    def get_data(self, data_path: str) -> pd.DataFrame:
        #Stores the file path in the class variable.
        self.data_path = data_path
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding='latin-1')
            
        print("Available columns:", df.columns.tolist())
        return df