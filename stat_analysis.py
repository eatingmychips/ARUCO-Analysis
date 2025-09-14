import csv 
import pandas as pd
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from scipy import signal, interpolate
from analysis import *
from plotting_funcs import *
import statistics as stat
import matplotlib.patches as mpatches
import math
import json
from os import listdir



######## Here we import the files necessary for analysis, we also import the representative files for gait plotting ########

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


#TODO: Enter your file path here:
file_path = r"L:\biorobotics\data\Vertical&InvertedClimbing\VerticalClimbingTrials\AllTrials"



files = [file_path + "\\" + x
            for x in find_csv_filenames(file_path)]


frequencies = [10, 20, 30, 40, 50]

def stat_analysis(files):
    # Declare empty lists to store data (optional if you want to store the results later)
   
    lateral_velocity = {}
    forward_velocity = {}
    body_angles = {}
    angular_velocity = {}
    

    dictionaries = [lateral_velocity, forward_velocity, body_angles, angular_velocity]
    #Declare trial_no variable
    turning_trial_no = 0
    elytra_trial_no = 0

    for file in files: 
        parts, stim_deets, stim_occur, fps = file_read(file)
        stim_dict = get_post_stim(parts, stim_deets, stim_occur, fps)

    
        for key, value in stim_dict.items(): 
            for pose_lst in value: 
                angles = [item[2] for item in pose_lst]
                angles = angle_interpolate(angles)
                #TODO: Clean up outlier trial

                
                
                pos = [[item[0], item[1]] for item in pose_lst]
                pos = pos_interpolate(pos)
                #Get middle and bottom points
                pos = remove_outliers_and_smooth(pos, alpha=0.2, z_thresh=2.5)
                angles = remove_outliers_and_smooth_1d(angles, alpha=0.2, z_thresh=2.5)

                #Get List of Body angles
                body_angle = get_body_angles(angles, fps)
                ang_vel = get_ang_vel(body_angle, fps)
                in_line_vel, transv_vel = body_vel(pos, angles, fps)

                for dict in dictionaries: 
                    if key not in dict: 
                        dict[key] = []
                if trial_is_outlier(body_angle, key): 
                    continue

                lateral_velocity[key].append(transv_vel)
                forward_velocity[key].append(in_line_vel)
                body_angles[key].append(body_angle)
                angular_velocity[key].append(ang_vel)
                
                if key[0] == "Both": 
                    elytra_trial_no += 1
                elif key[0] == "Right" or "Left": 
                    turning_trial_no += 1

    print("Number of Turning Trials is: ", turning_trial_no)
    print("Number of Forward Trials is: ", elytra_trial_no)
    
 
    return lateral_velocity, forward_velocity, body_angles, angular_velocity




lateral_velocity, forward_velocity, body_angles, angular_velocity = stat_analysis(files)

### CALL PLOTTNG FUNCTIONS ###
lateral_max, fwd_max, angles_max, ang_vel_max = get_max_values(lateral_velocity, forward_velocity, body_angles, angular_velocity)


# Call all time based plots
# antenna_time_plot(lateral_velocity, frequencies, 'Lateral velocity\n(mm/s)')
# antenna_time_plot(forward_velocity, frequencies, "Forward Velocity (units / s)")
# frequency_plot(angles_max, frequencies, "Angular Deviation (degrees)")

antenna_time_plot(body_angles, frequencies, "Angular Deviation (degrees)")
antenna_trials_plot(body_angles, frequencies, "Angular Deviation (degrees)")

# antenna_time_plot(angular_velocity, frequencies,  "Ang. Vel (deg/s)") 
# antenna_trials_plot(angular_velocity, frequencies,  "Ang. Vel (deg/s)")

# # Call all frequency based plots
# # frequency_plot(lateral_max, frequencies, "Lateral Velocity (units / s)")
# # frequency_plot(fwd_max, frequencies, "Forward Velocity (units / s)")
frequency_plot(angles_max, frequencies, "Angular Deviation (degrees)")
# frequency_plot(ang_vel_max, frequencies, "Ang. Vel (deg/s)")


# Call all time based plots
# elytra_time_plot(lateral_velocity, frequencies, "Lateral Velocity (mm/s)")
elytra_time_plot_single(forward_velocity, 30, "Induced Forward Velocity (mm/s)")
# elytra_time_plot(forward_velocity, frequencies, "Forward Velocity (mm/s)") 
elytra_trials_plot(forward_velocity, frequencies, "Forward Velocity (mm/s)")

# elytra_time_plot(body_angles, frequencies, "Body Angles (degs)")
# elytra_time_plot(body_angles, frequencies, "Ang. Vel. (degs/s)")

# frequency_plot_elytra(lateral_max, frequencies, "Lateral Velocity (units / s)")
frequency_plot_elytra(fwd_max, frequencies, "Forward Velocity (mm / s)")
# frequency_plot_elytra(angles_max, frequencies, "Angular Deviation (degrees)")
# frequency_plot_elytra(ang_vel_max, frequencies, "Ang. Vel (deg/s)")


