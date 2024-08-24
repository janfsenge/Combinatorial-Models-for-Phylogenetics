""" Collection of auxiliary functions and classes to be used in the project.

created by: Jan F Senge, last change 07.08.2024
"""

# %%
# import packages
# Numpy change it's output behavior;
# now it prints the type like np.float(0.0)
import numpy as np
# np.set_printoptions(legacy="1.25")

import pandas as pd
from pathlib import Path
import time

# %%
# Timer class

# to make it a bit easier we use our own timer class
# and we will use the perf_counter_ns and process_time_ns

class OwnTimer:
    """MixMax of code to make a timer class to 
    compare perfomance of different parts of the code.
    """
    def __init__(self, num_timers=1, names=None,
                 return_seconds=False,
                 additional_infocols=None):
        if num_timers is None:
            if names is None:
                raise ValueError('Either num_timers or names must be provided.')
            else:
                num_timers = len(names)
        
        if names is None:
            self.names = np.arange(num_timers)
        else:
            if num_timers == len(names):
                self.names = names
            else:
                num_timers = len(names)
                self.names = names

        self.buckets = np.zeros([num_timers, 2])
        self.counter = np.zeros(num_timers)   
        self.running_timers = {}
        self.return_seconds = return_seconds

        self.metadata = None
        if additional_infocols is not None:
            if (isinstance(additional_infocols, list)
                    and isinstance(additional_infocols[0], str)):
                pass
            else:
                raise ValueError('additional_infocols must be a list of strings.')

            self.metadata = [{x: None for x in additional_infocols}
                             for _ in range(num_timers)]

        self.states = None
        return None

    def save_with_information(self, additional_info,
                              additional_info_names,
                              reset_times=True,
                              reset_counters=True):

        df = self.get_dataframe()
        additional_info = np.array(additional_info).reshape(1, -1)
        if isinstance(additional_info_names, list):
            if len(additional_info_names) != np.shape(additional_info)[1]:
                raise ValueError('additional info and names do not fit!')

        if len(additional_info) == 1:
            df.loc[:, additional_info_names] = additional_info[0]
        else:
            df.loc[:, additional_info_names] = additional_info

        if self.states is None:
            self.states = df.copy()
        else:
            self.states = pd.concat([self.states, df], axis=0).reset_index(drop=True).copy()

        col_select = [col for col in self.states.columns if col != 'index']
        self.states[col_select] = self.states[col_select].astype(np.int64)

        if reset_times:
            self.reset_times(reset_states=False, reset_counters=reset_counters)

        # return None

    def start(self, id=0):
        self.running_timers[id] = [time.perf_counter_ns(), time.process_time_ns()]
        return self

    def stop(self, id=0):
        end = [time.perf_counter_ns(), time.process_time_ns()]
        if id not in self.running_timers:
            raise ValueError('Timer not started yet.')

        if isinstance(id, str):
            if id in self.names:
                id_i = self.names.index(id)
            else:
                raise ValueError('Timer name not found.')
        else:
            id_i = int(id)
        
        self.buckets[id_i, :] += np.array(end) - np.array(self.running_timers[id])
        self.counter[id_i] += 1

        del self.running_timers[id]
        return self
    
    def end(self, id=0):
        self.stop(id)
        return self
    
    def add_infotocounter(self, id=0, infotoadd=None):
        if infotoadd is None:
            return self
        if isinstance(id, str):
            if id in self.names:
                id_i = self.names.index(id)
            else:
                raise ValueError('Timer name not found.')
        else:
            id_i = int(id)

        if isinstance(infotoadd, dict):
            for key, value in infotoadd.items():
                if key in self.metadata[id_i]:
                    self.metadata[id_i][key] = value
                else:
                    print(f'Key {key} not found in metadata.')
        elif isinstance(infotoadd, list):
            raise ValueError('to add to metadata, infotoadd must be a dictionary (not a list).')
        elif len(self.metadata[id_i].keys()) == 1:
            key = list(self.metadata[id_i].keys())[0]
            self.metadata[id_i][key] = \
                infotoadd
        else:
            raise ValueError('to add to metadata, infotoadd must be a dictionary,'
                             'or the metadata must have only one key.')

        return self
    
    def get_metadata(self):
        df = pd.DataFrame(self.metadata)
        df.loc[:, 'index'] = self.names

        return df
    
    def reset_times(self, reset_states=False, reset_counters=True):
        self.buckets = np.zeros_like(self.buckets)
        if reset_counters:
            self.counter = np.zeros_like(self.counter)
        self.running_timers = {}

        if reset_states:
            self.states = None
        return self

    def show_timers(self, return_seconds=None):
        if return_seconds is None:
            return_seconds = self.return_seconds

        print('Timers:')
        for i, name in enumerate(self.names):
            if self.counter[i] == 0:
                continue

            print('------------------------')
            print(f'{name}:')
            if return_seconds:
                print('  total:', f'{self.buckets[i, 0]/1e9 :.5f}s, {self.buckets[i, 1]/1e9:.5f}s')
                if self.counter[i] > 1:
                    print('  mean: ', f'{self.buckets[i, 0]/1e9/self.counter[i]:.5f}s, {self.buckets[i, 1]/1e9/self.counter[i]:.5f}s')
            else:
                print('  total:', f'{self.buckets[i, 0] :.0f}ns, {self.buckets[i, 1]:.0f}ns')
                if self.counter[i] > 1:
                    print('  mean: ', f'{self.buckets[i, 0]/self.counter[i]:.0f}ns, {self.buckets[i, 1]/self.counter[i]:.0f}ns')
        return ''

    def get_dataframe(self, return_seconds=None):
        if return_seconds is None:
            return_seconds = self.return_seconds

        if return_seconds:
            buckets = self.buckets/1e9
        else:
            buckets = self.buckets

        df = pd.DataFrame(data=np.hstack([buckets, self.counter.reshape(-1, 1)]),
                index=self.names,
                columns=['perf_counter', 'process_time', 'counter']).reset_index(drop=False)
        if self.metadata is not None:
            df2 = self.get_metadata()
            df = pd.merge(df, df2, on='index', how='left')
        return df

    def get_state(self, return_seconds=None):
        if return_seconds is None:
            return_seconds = self.return_seconds

        if self.states is None:
            raise ValueError('No states saved yet.')
        
        df = self.states.copy()
        if return_seconds:
            df = df.astype({'perf_counter': 'float64', 'process_time': 'float64'})
            df.loc[:, 'perf_counter'] = df.loc[:, 'perf_counter']/1e9
            df.loc[:, 'process_time'] = df.loc[:, 'process_time']/1e9
        return df
