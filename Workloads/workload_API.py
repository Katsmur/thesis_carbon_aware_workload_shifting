import pandas as pd
from abc import ABC, abstractmethod


class Job(ABC):

    @abstractmethod
    def origin_location(self):
        pass
    
    @abstractmethod
    def start_time(self):
        pass

    @abstractmethod
    def duration(self):
        pass


class Job_DF(pd.DataFrame):

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self._validate(self)
        
    @staticmethod
    def _validate(self):
        expected_columns = set(['start_timestamp', 'duration', 'location', 'num_of_requests'])
        actual_columns = set(self.columns)
        missing_cols = expected_columns - actual_columns
        
        if missing_cols:
            raise AttributeError(f"Columns {missing_cols} are missing in the DataFrame")
        d1 = self.dtypes.astype(str).to_dict()
        if d1['start_timestamp'] != 'datetime64[ns]':
            raise AttributeError(f"start_timestamp must be of dtype datetime64[ns], is {d1['start_timestamp']}")
        if d1['duration'] != 'timedelta64[ns]':
            raise AttributeError(f"duration must be of dtype timedelta64[ns], is {d1['duration']}")
        if d1['num_of_requests'] != 'int64' and d1['num_of_requests'] != 'float64':
            raise AttributeError(f"num_of_requests must be either dtype int64 or float64, is {d1['num_of_requests']}")
        
        self = self.sort_values(by='start_timestamp')

    def to_df(self):
        """to get dataframe in pd.DataFrame format"""
        return pd.DataFrame(self.values, columns=self.columns, index=self.index)
        
    def to_base_list(self):
        """to get list of basic columns"""
        df = pd.DataFrame(self.values, columns=self.columns, index=self.index)

        return df[['start_timestamp', 'duration', 'location', 'num_of_requests']].values.tolist()