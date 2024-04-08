import pandas as pd
import numpy as np
from collections import defaultdict

class DataPreprocess():
    '''
    A class for preprocessing data for Markov Chain analysis.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - col_id (str): The name of the column containing user IDs.
    - col_time (str): The name of the column containing timestamps.
    - col_chan (str): The name of the column containing channel information.
    - col_conv (str): The name of the column indicating conversion events.

    Methods:
    - preprocess_data(): Preprocesses the input data to create a DataFrame suitable for Markov Chain analysis.
    '''
    def __init__(self, df: pd.DataFrame, col_id: str, col_time: str, col_chan: str, col_conv: str):
        self.df = df 
        self.col_id = col_id
        self.col_time = col_time
        self.col_chan = col_chan
        self.col_conv = col_conv


    def preprocess_data(self):
        '''
        Preprocesses the input data to create a DataFrame suitable for Markov Chain analysis.

        Returns:
        - df_paths (pd.DataFrame): A DataFrame containing preprocessed data with user paths.
        '''
        df = self.df
        col_id = self.col_id
        col_time = self.col_time
        col_chan = self.col_chan
        col_conv = self.col_conv
        
        df_markov = df.copy(deep=True)

        df_markov = df_markov.sort_values([col_id, col_time])
        df_markov['state_pos'] = df_markov.groupby(col_id).cumcount() + 1

        # create df_conversion, showing whether a user convert or not convert
        df_conversion = df_markov.drop_duplicates(col_id, keep='last')[[col_id, col_conv]]

        # groupby cookie and concat all strings as 'paths' column to show the path a user went through
        df_paths = df_markov.groupby(col_id).agg(paths=(col_chan, lambda x: ', '.join(map(str, x))))

        # merge with df_conversion and reset index
        df_paths = pd.merge(df_conversion, df_paths, left_on=col_id, right_index=True)
        df_paths = df_paths.reset_index(drop=True)

        # concat "start" at the beginning of the path
        # if no conversion made, we append "no_conv", otherwise "conv"
        df_paths['paths'] = np.where(
            df_paths[col_conv] == 0,
            'Start, ' + df_paths['paths'] + ', No_Conv',
            'Start, ' + df_paths['paths'] + ', Conv'
        )

        # finalize the 'paths' column by converting it into a list
        df_paths['paths'] = df_paths['paths'].str.split(', ')

        return df_paths


class MarkovChainAnalysis():
    '''
    A class for performing Markov Chain analysis on preprocessed data.

    Parameters:
    - df_paths (pd.DataFrame): The preprocessed DataFrame containing user paths.

    Methods:
    - create_transition_matrix(): Creates a transition matrix based on the user paths.
    - removal_effects(): Calculates the removal effects of each channel.
    - attribution_total(): Calculates the total attribution for each channel.
    - removal_attribution_channels(): Combines removal effects and attributions into a single DataFrame.

    '''
    def __init__(self, df_paths):
        self.df_paths = df_paths

    
    def create_transition_matrix(self):
        '''
        Creates a transition matrix based on the user paths.

        Returns:
        - transition_matrix (pd.DataFrame): The transition matrix.
        '''
        df_paths = self.df_paths

        paths_list = df_paths['paths']
        total_conv = sum(path.count('Conv') for path in df_paths['paths'])
        conv_rate = total_conv / len(paths_list)

        # make a set of all unique channel list for creating transition states
        unique_channel_list = set(x for element in paths_list for x in element)

        # create transition states from the unique channel list, and assign the value with 0 first
        transition_states = {x + '>' + y: 0 for x in unique_channel_list for y in unique_channel_list}

        for possible_state in unique_channel_list:
            # exclude conversion / no_conversion values from the calculations.
            if possible_state not in ['Conv', 'No_Conv']:
                # iterates over the path list we have
                for user_path in paths_list:
                    # checks if the possible_state is present in the current user_path
                    if possible_state in user_path:
                        # creates a list indices that contains the indices of the user_path where possible_state occurs.
                        indices = [i for i, s in enumerate(user_path) if possible_state in s]
                        for col in indices:
                            # updates the transition_states dictionary. 
                            # extracts the current channel (user_path[col]) and the next channel (user_path[col + 1]) from the user_path, 
                            # concatenates them with '>', and uses the result as a key in the transition_states dictionary. 
                            # then increments the value associated with that key by 1, indicating the transition from the current channel to the next channel.
                            transition_states[user_path[col] + '>' + user_path[col + 1]] += 1
        
        transition_prob = defaultdict(dict)

        for state in unique_channel_list:
            # exclude conversion and no_conversion state from the calculation
            if state not in ['Conv', 'No_Conv']:
                # counter, used to calculate the total number of transitions involving the current state.
                counter = 0
                index = [i for i, s in enumerate(transition_states) if state + '>' in s]
                
                # first iteration, to count the number of transitions for the current transition state
                for col in index:
                    # if the transition count for the current transition state is greater than 0
                    # increments the counter by the number of transitions for the current transition state
                    if transition_states[list(transition_states)[col]] > 0:
                        counter += transition_states[list(transition_states)[col]]
                
                # restart the iteration, to keep the counter 
                for col in index:
                    # calculates the transition probability for the current transition state
                    # it divides the number of transitions for the current state by the total number of transitions involving the current state
                    if transition_states[list(transition_states)[col]] > 0:
                        state_prob = float((transition_states[list(transition_states)[col]])) / float(counter)
                        transition_prob[list(transition_states)[col]] = state_prob
        
        transition_matrix = pd.DataFrame()
        list_of_unique_channels = set(x for element in paths_list for x in element)

        # assign zero to all matrix elements
        # it also sets the diagonal element for each channel to 1.0 if the channel is 'Conv' or 'No_Conv', indicating that a conversion event occurs
        for channel in list_of_unique_channels:
            transition_matrix[channel] = 0.00
            transition_matrix.loc[channel] = 0.00
            transition_matrix.loc[channel][channel] = 1.0 if channel in ['Conv', 'No_Conv'] else 0.0

        # assign probability using calculated transition probability
        for key, value in transition_prob.items():
                origin, destination = key.split('>')
                transition_matrix.at[origin, destination] = value

        # reorder the transition matrix so it's always start with "Start" state and end with "Conv" and "No_Conv" state
        desired_order = ['Start'] + [col for col in transition_matrix.columns if col not in ['Start', 'Conv', 'No_Conv']] + ['Conv', 'No_Conv']
        transition_matrix = transition_matrix.reindex(columns=desired_order, index=desired_order)

        self.transition_matrix = transition_matrix
        self.conv_rate = conv_rate
        self.total_conv = total_conv

        return transition_matrix
    

    def removal_effects(self):
        '''
        Calculates the removal effects of each channel.

        Returns:
        - removal_effects_dict (dict): A dictionary containing removal effects for each channel.
        '''
        transition_matrix = self.transition_matrix
        conv_rate = self.conv_rate

        removal_effects_dict = {}

        # creates a list channels containing all channels from transition_matrix 
        # except for 'Start', 'No_Conv', and 'Conv'
        channels = [channel for channel in transition_matrix.columns if channel not in ['Start',
                                                                                        'No_Conv',
                                                                                        'Conv']]

        # this loop iterates over each channel in channels and calculates its removal effect
        for channel in channels:
            removal_df = transition_matrix.drop(channel, axis=1).drop(channel, axis=0)
            
            for column in removal_df.columns:

                # calculates the sum of values in each row of removal_df 
                # computes the percentage of null transitions (transitions to 'No_Conv'), and updates the 'No_Conv' column in removal_df with the null percentages.
                row_sum = np.sum(list(removal_df.loc[column]))
                null_pct = float(1) - row_sum
                if null_pct != 0:
                    removal_df.loc[column]['No_Conv'] = null_pct
                removal_df.loc['No_Conv']['No_Conv'] = 1.0
            
            # calculates the removal effect for the current channel by performing matrix operations using numpy and pandas 
            # to simulate the removal of the channel and its effects on conversion rates.
            removal_to_conv = removal_df[['No_Conv', 'Conv']].drop(['No_Conv', 'Conv'], axis=0)
            removal_to_non_conv = removal_df.drop(['No_Conv', 'Conv'], axis=1).drop(['No_Conv', 'Conv'], axis=0)
            removal_inv_diff = np.linalg.inv(
                np.identity(
                    len(removal_to_non_conv.columns)) - np.asarray(removal_to_non_conv))
            
            removal_dot_prod = np.dot(removal_inv_diff, np.asarray(removal_to_conv))
            removal_cvr = pd.DataFrame(removal_dot_prod,
                                    index=removal_to_conv.index)[[1]].loc['Start'].values[0]
            
            removal_effect = 1 - removal_cvr / conv_rate

            # the removal effect for the current channel is stored in the removal_effects_dict dictionary.
            removal_effects_dict[channel] = removal_effect

        self.removal_effects_dict = removal_effects_dict

        return removal_effects_dict
    

    def attribution_total(self):
        '''
        Calculates the total attribution for each channel.

        Returns:
        - attribution_dict (dict): A dictionary containing total attribution for each channel.
        '''
        total_conv = self.total_conv
        removal_effects_dict = self.removal_effects_dict

        removal_effect_sum = np.sum(list(removal_effects_dict.values()))

        # calculates the attribution for each channel. It iterates over each key-value pair in removal_effects_dict, 
        # divides the removal effect value by the total removal effect sum, 
        # and then multiplies by the total conversions (total_conv) to get the attribution for that channel

        attribution_dict = {key: (value / removal_effect_sum) * total_conv for key, value in removal_effects_dict.items()}

        self.attribution_dict = attribution_dict

        return attribution_dict


    def removal_attribution_channels(self):
        '''
        Combines removal effects and attributions into a single DataFrame.

        Returns:
        - att_markov (pd.DataFrame): A DataFrame containing removal effects and attributions for each channel.
        '''
        removal_effects_dict = self.removal_effects()
        attribution_dict = self.attribution_total()

        removal_df = pd.DataFrame.from_dict(removal_effects_dict, orient='index', columns=['removal_effect'])
        att_df = pd.DataFrame.from_dict(attribution_dict, orient='index', columns=['attribution'])

        # combine dataframes
        att_markov = pd.concat([removal_df, att_df], axis=1)
        att_markov = round(att_markov, 2)

        return att_markov
