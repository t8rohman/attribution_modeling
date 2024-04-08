import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SimpleModels():
    '''
    A class for simple attribution models.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col_id (str): The column name for the user ID.
    - col_time (str): The column name for the timestamp of the interaction.
    - col_chan (str): The column name for the marketing channel.
    - col_conv (str): The column name for the conversion event (1 if conversion, 0 otherwise).
    '''

    def __init__(self, df: pd.DataFrame, col_id: str, col_time: str, col_chan: str, col_conv: str):
        self.df = df 
        self.col_id = col_id
        self.col_time = col_time
        self.col_chan = col_chan
        self.col_conv = col_conv


    def first_touch(self, to_return='df'):
        '''
        Calculate first-touch attribution.

        Parameters:
        - to_return (str): Return type ('df' for DataFrame or 'plot' for plot).

        Returns:
        - DataFrame or plot based on the `to_return` parameter.
        '''
        df = self.df
        col_id = self.col_id
        col_time = self.col_time
        col_chan = self.col_chan
        col_conv = self.col_conv

        # only take data where the user make a conversion

        df = df.copy(deep=True)
        
        cookie_conv = df[df[col_conv] == 1][col_id].unique()
        df = df[df[col_id].isin(cookie_conv)]

        # check the first touch point of the user with our campaign, then filter out the data to only include these users

        df['min_time'] = df.groupby(col_id).transform('min')[col_time]
        df = df[df[col_time] == df['min_time']]

        att_first = pd.DataFrame(df[col_chan].value_counts())

        if to_return == 'df':
            return att_first
        
        elif to_return == 'plot':

            att_first.plot(kind='bar', rot=45)

            plt.title('First-touch Attribution')
            return plt.show()
        
        else:
            raise ValueError("to_return should be 'df' or 'plot'")
        
    
    def last_touch(self, to_return='df'):
        '''
        Calculate last-touch attribution.

        Parameters:
        - to_return (str): Return type ('df' for DataFrame or 'plot' for plot).

        Returns:
        - DataFrame or plot based on the `to_return` parameter.
        '''
        df = self.df
        col_id = self.col_id
        col_time = self.col_time
        col_chan = self.col_chan
        col_conv = self.col_conv
        
        # only take data where the user make a conversion

        df = df.copy(deep=True)

        cookie_conv = df[df[col_conv] == 1][col_id].unique()
        df = df[df[col_id].isin(cookie_conv)]

        # check the last touch point of the user with our campaign, then filter out the data to only include these users

        df['max_time'] = df.groupby(col_id).transform('max')[col_time]
        df = df[df[col_time] == df['max_time']]

        att_last = pd.DataFrame(df[col_chan].value_counts())

        if to_return == 'df':
            return att_last
        
        elif to_return == 'plot':

            att_last.plot(kind='bar', rot=45)

            plt.title('Last-touch Attribution')
            return plt.show()
        
        else:
            raise ValueError("to_return should be 'df' or 'plot'")


    def linear_touch(self, to_return='df'):
        '''
        Calculate linear-touch attribution.

        Parameters:
        - to_return (str): Return type ('df' for DataFrame or 'plot' for plot).

        Returns:
        - DataFrame or plot based on the `to_return` parameter.
        '''
        df = self.df
        col_id = self.col_id
        col_time = self.col_time
        col_chan = self.col_chan
        col_conv = self.col_conv

        df = df.copy(deep=True)

        cookie_conv = df[df[col_conv] == 1][col_id].unique()
        df = df[df[col_id].isin(cookie_conv)]

        df = df.sort_values([col_id, col_time]).reset_index(drop=True)

        # check number of touchpoints before converting, and put it as denominator, as the credits are distributed equally

        df['n_touchpoints'] = df.groupby(col_id).transform('size')
        df['credits'] = 1 / df['n_touchpoints']

        att_linear = pd.DataFrame(df.groupby(col_chan)['credits'].sum().sort_values(ascending=False))

        if to_return == 'df':
            return att_linear
        
        elif to_return == 'plot':
            att_linear.plot(kind='bar')

            plt.title('Linear Attribution')
            plt.show()
        
        else:
            raise ValueError("to_return should be 'df' or 'plot'")

    
    def time_decay(self, lambda_val=0.0001, to_return='df'):
        '''
        Calculate time-decay attribution.

        Parameters:
        - lambda_val (float): The decay factor, the higher the faster channel will decay.
        - to_return (str): Return type ('df' for DataFrame or 'plot' for plot).

        Returns:
        - DataFrame or plot based on the `to_return` parameter.
        '''
        df = self.df
        col_id = self.col_id
        col_time = self.col_time
        col_chan = self.col_chan
        col_conv = self.col_conv

        df = df.copy(deep=True)

        cookie_conv = df[df[col_conv] == 1][col_id].unique()
        df = df[df[col_id].isin(cookie_conv)]

        df = df.sort_values([col_id, col_time]).reset_index(drop=True)

        # check the maximum time, to know when the person made a conversion
        df['conv_time'] = df.groupby(col_id)[col_time].transform('max')

        # check the different time, to know the proximity to the conversion
        # and convert it to minute
        df['diff_time'] = df['conv_time'] - df[col_time]
        df['diff_mins'] = df['diff_time'].dt.total_seconds() // 60

        # calculate the credits by using the formula above
        df['credits'] = np.exp(-lambda_val * df['diff_mins'])

        # normalize the credits to make it added up to 1
        sum_creds = df.groupby(col_id)['credits'].transform('sum')
        df['normalized_credits'] = df['credits'] / sum_creds

        att_time = pd.DataFrame(df.groupby(col_chan)['normalized_credits'].sum().sort_values(ascending=False))

        if to_return == 'df':
            return att_time
        
        elif to_return == 'plot':
            att_time.plot(kind='bar', rot=45)

            plt.title('Time-decay Attribution')
            plt.show()

        else:
            raise ValueError("to_return should be 'df' or 'plot'")
        

    def position_based(self, fl_weight=0.4, to_return='df'):
        '''
        Calculate position-based attribution.

        Parameters:
        - fl_weight (float): The weight for first and last touch.
        - to_return (str): Return type ('df' for DataFrame or 'plot' for plot).

        Returns:
        - DataFrame or plot based on the `to_return` parameter.
        '''
        df = self.df
        col_id = self.col_id
        col_time = self.col_time
        col_chan = self.col_chan
        col_conv = self.col_conv

        df = df.copy(deep=True)

        cookie_conv = df[df[col_conv] == 1][col_id].unique()
        df = df[df[col_id].isin(cookie_conv)]

        df = df.sort_values([col_id, col_time]).reset_index(drop=True)

        df['n_touchpoints'] = df.groupby(col_id).transform('size')
        df['pos_touchpoints'] = df.groupby(col_id).cumcount() + 1

        for i in range(len(df)):
            if df.loc[i, 'n_touchpoints'] == 1:
                df.loc[i, 'credits'] = 1
            elif df.loc[i, 'n_touchpoints'] == 2:
                df.loc[i, 'credits'] = 0.5
            else:
                if df.loc[i, 'pos_touchpoints'] == 1 or df.loc[i, 'pos_touchpoints'] == df.loc[i, 'n_touchpoints']:
                    df.loc[i, 'credits'] = fl_weight
                else:
                    df.loc[i, 'credits'] = (1 - (fl_weight * 2)) / (df.loc[i, 'n_touchpoints'] - 2)

        att_pos = pd.DataFrame(df.groupby(col_chan)['credits'].sum().sort_values(ascending=False))
        
        if to_return == 'df':
            return att_pos
        
        elif to_return == 'plot':

            att_pos.plot(kind='bar', rot=45)

            plt.title('Position-Based Attribution')
            plt.show()

        else:
            raise ValueError("to_return should be 'df' or 'plot'")
        
    
    def combine_all_methods(self, time_decay_lambda=0.0001, position_based_fl_weight=0.4):
        '''
        Combine all attribution methods into a single DataFrame.

        Parameters:
        - time_decay_lambda (float): The decay factor for time-decay attribution.
        - position_based_fl_weight (float): The weight for first and last touch in position-based attribution.

        Returns:
        - DataFrame containing the combined attribution results.
        '''
        att_first = self.first_touch(to_return='df')
        att_last = self.last_touch(to_return='df')
        att_linear = self.linear_touch(to_return='df')
        att_time = self.time_decay(time_decay_lambda, to_return='df')
        att_pos = self.position_based(position_based_fl_weight, to_return='df')

        # combine all the DataFrames
        att_all = pd.concat([att_first, att_last, att_linear, att_time, att_pos], 
                            axis=1,
                            keys=['att_first', 'att_last', 'att_linear', 'att_time', 'att_pos'])

        att_all.columns = att_all.columns.droplevel(1)
        att_all = round(att_all, 2)

        return att_all 
