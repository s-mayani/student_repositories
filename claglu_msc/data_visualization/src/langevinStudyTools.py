import os
import sys
import typing

import numpy as np
import pandas as pd

# Utility functions used in `langevin_study/emittance.ipynb`

# Correction to use nanometer as unit for normalized emittance [cm] -> [nm]
def scale_emittance(df_dict):
    for experiment, file_path in df_dict.items():
        df_dict[experiment]['NepsX'] *= 1e7
        df_dict[experiment]['NepsY'] *= 1e7
        df_dict[experiment]['NepsZ'] *= 1e7
    return df_dict

# Scale Statistics to SI units
def scale_p3m(df_dict):
    for experiment, file_path in df_dict.items():
                        
        # Scale rrmsX [cm] -> m
        df_dict[experiment]['rrmsX'] /= 1e2
        df_dict[experiment]['rmeanX'] /= 1e2
        # Scale <vmean>^2 [(cm/ms)^2] -> m
        df_dict[experiment]['vmeanX'] /= 1e2
        
        # Scale Temparature [(cm/s)^2] -> [(m/s)^2]
        if 'Tx' in df_dict[experiment].columns:
            df_dict[experiment]['Tx'] /= 1e4
            df_dict[experiment]['Ty'] /= 1e4
            df_dict[experiment]['Tz'] /= 1e4
        
        EField_factor = 1e-2
        if 'avgEfield_x' in df_dict[experiment].columns:
            # Scale Efield to [m m_e s^{-2} q_e^{-1}]
            df_dict[experiment]['avgEfield_x'] *= EField_factor
            df_dict[experiment]['avgEfield_y'] *= EField_factor
            df_dict[experiment]['avgEfield_z'] *= EField_factor
            # # [m m_e s^{-2} q_e^{-1}] -> [m kg s^{-2} q_e^{-1}]
            # avgEF_df.iloc[:,-3:] *= 9.10938e31
            # # [m kg s^{-2} q_e^{-1}] -> [m kg s^{-2} C^{-1}]
            # avgEF_df.iloc[:,-3:] *= 1.602e-19

            # [m m_e s^{-2} q_e^{-1}] -> [m kg s^{-2} C^{-1}]
            # avgEF_df.iloc[:,-3:] /= 5.685e-14
        
        if 'avgEfield_particle_x' in df_dict[experiment].columns:
            # Scale Efield to [m m_e s^{-2} q_e^{-1}]
            df_dict[experiment]['avgEfield_particle_x'] *= EField_factor
            df_dict[experiment]['avgEfield_particle_y'] *= EField_factor
            df_dict[experiment]['avgEfield_particle_z'] *= EField_factor
            
        return df_dict

def load_p3m_data(p3m_root_dir = "p3m_data"):

    # Get all directories
    dir_generator = next(os.walk (p3m_root_dir))[1]
    # Get all Experiment directories
    experiment_paths = [d for d in dir_generator if d.startswith("rc")]
    # Extract experiment key
    experiment_names = [d.replace('_', '=') for d in experiment_paths]

    # Create dict containing each experiment with folder name as key
    experiment_dict = { k:v for k, v in zip(experiment_names, experiment_paths)}

    # Loading all dataframes
    df_dict = {}
    for experiment, file_path in experiment_dict.items():
        print("Loading `{}`".format(experiment))
        curr_df = pd.read_csv("/".join([p3m_root_dir, file_path, 'data', 'BeamStatistics.csv']))
                
        # Add  Temperature columns from `Temperature.csv`
        try:
            curr_df = pd.merge(curr_df, pd.read_csv("/".join([p3m_root_dir, file_path, 'data', 'Temperature.csv'])))
            curr_df = curr_df.rename(columns={'temp_x' : 'Tx', 'temp_y' : 'Ty', 'temp_z' : 'Tz', })
        except Exception as error:
            pass
        
        # Add `time` column with dt = 2.15623e–13s
        curr_df.insert(1, 'time', curr_df['it'].apply(lambda x: float(x)*2.15623e-13)) # [s]
        
        # Add plasma period
        curr_df.insert(2, 'tau_p', curr_df['time'].apply(lambda x: float(x)/4.3114e-11)) # [s]
        
        # Scale emittance accordingly to get Normalized Emittance
        # norm_emittance_factor = 10e-3 * 1.0/29979245800.0
        
        # Divide by light speed [cm, s] to obtain normalized emittance.
        # Can be done since \gamma \approx 1.0
        norm_emittance_factor = 1.0/29979245800.0
        
        curr_df['NepsX'] = curr_df['epsX'] * norm_emittance_factor
        curr_df['NepsY'] = curr_df['epsY'] * norm_emittance_factor
        curr_df['NepsZ'] = curr_df['epsZ'] * norm_emittance_factor

        # Add Avg Efield if `AvgEfield.csv` is present
        try:
            avgEF_df = pd.read_csv("/".join([p3m_root_dir, file_path, 'data', 'AvgEfield.csv']))
            
            # Join on 'it'
            curr_df = pd.merge(curr_df, avgEF_df, how='inner', on='it')

        except Exception as error:
            pass
        
        df_dict[experiment] = curr_df
    
    return df_dict

# Scale statistics to SI units
def scale_langevin(df_dict):
    for experiment, file_path in df_dict.items():
        # Only do this for the X-components
        df_dict[experiment]['time'] /= 1e3 # [ms] -> [s]
        df_dict[experiment]['rrmsX'] /= 1e2 # [cm] -> [m]
        df_dict[experiment]['rmeanX'] /= 1e2 # [cm] -> [m]
        df_dict[experiment]['epsX'] /= 1e1 # [cm^2 / ms] -> [m^2/s]
        df_dict[experiment]['NepsX'] /= 1e2 # [cm] -> [nm]
        df_dict[experiment][['vmaxX','vminX','vmeanX']] *= 10 # [cm/ms] -> [m/s]
        df_dict[experiment][['rmaxX','rminX','rmeanX']] /= 100 # [cm] -> [m]
        df_dict[experiment][['Tx']] /= 1e4 # [(cm/ms)^2] -> [m/s)^2]
        if 'avgEfield_x' in df_dict[experiment].columns: 
            # [cm m_e ms^{-2} q_e^{-1}] -> [m m_e s^{-2} q_e^{-1}]
            df_dict[experiment][['avgEfield_x']] *= 1e4
            df_dict[experiment][['avgEfield_y']] *= 1e4
            df_dict[experiment][['avgEfield_z']] *= 1e4
            # # [m m_e s^{-2} q_e^{-1}] -> [m kg s^{-2} q_e^{-1}]
            # df_dict[experiment][['avgEfield_x']] *= 9.10938e31
            # df_dict[experiment][['avgEfield_y']] *= 9.10938e31
            # df_dict[experiment][['avgEfield_z']] *= 9.10938e31
            # # [m m_e s^{-2} q_e^{-1}] -> [m kg s^{-2} C^{-1}]
            # df_dict[experiment][['avgEfield_x']] *= 1.602e-19
            # df_dict[experiment][['avgEfield_y']] *= 1.602e-19
            # df_dict[experiment][['avgEfield_z']] *= 1.602e-19
            
            # [cm m_e ms^{-2} q_e^{-1}] -> [m kg s^{-2} C^{-1}]
            # df_dict[experiment][['avgEfield_x']] /= 5.685e-8
            # df_dict[experiment][['avgEfield_y']] /= 5.685e-8
            # df_dict[experiment][['avgEfield_z']] /= 5.685e-8
    
    return df_dict


def load_langevin_data(langevin_root_dir="langevin_data", scale=True):

    # Get all directories
    dir_generator = next(os.walk (langevin_root_dir))[1]
    # Get all Experiment directories
    experiment_paths = [d for d in dir_generator if d.startswith("langevin")]
    # Extract experiment key
    experiment_names = ["_".join(d.split('_')[1:-2]) for d in experiment_paths]
    
    # Create dict containing each experiment with folder name as key
    experiment_dict = { k:v for k, v in zip(experiment_names, experiment_paths)}
    
    # Loading all dataframes
    df_dict = {}
    for experiment, file_path in experiment_dict.items():
        print("Loading `{}`".format(experiment))
        df_dict[experiment] = pd.read_csv("/".join([langevin_root_dir, file_path, 'All_FieldLangevin_0.csv']))
                
        # Add plasma period (divide by plasma frequency)
        df_dict[experiment].insert(2, 'tau_p', df_dict[experiment]['time'].apply(lambda x: float(x)/4.3114e-11)) # [s]

        # Add Normalized emittance if not present from `FieldLangevin_1.csv`
        if 'NepsX' not in df_dict[experiment].columns:
            df_dict[experiment]['NepsX'] = pd.Series(pd.read_csv("/".join([langevin_root_dir, file_path, 'FieldLangevin_0.csv']))['Neps_X'])
        
    return df_dict

# Load a directory that contains experiments launched via an array submission
def load_array_langevin_data(langevin_array_root_dir):
    # Get all directories
    dir_generator = next(os.walk(langevin_array_root_dir))[1]
    # Get all Experiment directories
    experiment_paths = [d for d in dir_generator]
    # Extract experiment key
    experiment_names = [d.replace('_', '=') for d in experiment_paths]    
    
    # Create dict containing each experiment with folder name as key
    experiment_dict = { k:v for k, v in zip(experiment_names, experiment_paths)}
    
    # Loading all dataframes
    df_dict = {}
    for experiment, file_path in experiment_dict.items():
        print("Loading `{}`".format(experiment))
        full_path = "/".join([langevin_array_root_dir, file_path])
        data_dir = [d for d in next(os.walk(full_path))[1] if d.startswith('langevin')][0]
        
        df_dict[experiment] = pd.read_csv("/".join([full_path, data_dir, 'All_FieldLangevin_0.csv']), index_col=False)
                
        # Add plasma period
        df_dict[experiment].insert(2, 'tau_p', df_dict[experiment]['time'].apply(lambda x: float(x)/4.3114e-11)) # [s]
        
        # Add Normalized emittance if not present from `FieldLangevin_1.csv`
        if 'NepsX' not in df_dict[experiment].columns:
            df_dict[experiment]['NepsX'] = pd.Series(pd.read_csv("/".join([langevin_array_root_dir, file_path, 'FieldLangevin_0.csv']))['Neps_X'])
        
    
    return df_dict

# Clip Rows to certain time range (s)
def clip_time_df(df : pd.core.frame.DataFrame, t_min=0.0, t_max=2.2e-10):
    return df[(df['time'] >= t_min) & (df['time'] <= t_max)]

def clip_time_dict(df_dict : typing.Dict[str, pd.core.frame.DataFrame], t_min=0.0, t_max=2.2e-10):
    for key, experiment_df in df_dict.items():
        df_dict[key] = clip_time_df(experiment_df, t_min, t_max)
    return df_dict