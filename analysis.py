import pandas as pd
import itertools as it
import numpy as np
from scipy import stats
import ast
from os import listdir
import math

def file_read(file):
    """Reads in a single csv file, with 3 columns: time, pose (position of the insect, structured as 
    [top, middle, bottom]), and finally arduino data (stimulation side, frequency 
    of the stimulation and duration of the stimulation) """


    df = pd.read_csv(file)
    # Read in time, pose and arduino data. 
    time = df.get('time')
    differences = time.diff()
    fps = 1/differences.mean()


    pose_raw = df.get('pose')
    arduino_data = df.get('arduino_data')
    stim_deets = []     # Will contain a list of lists: [stimulation side, frequency] 
                        # or [None, None] if no stimulation has occured.
    stim_occur = []     #Simply a binary list denoting whether a stimulation has occured at specific time j. 

    pose = []
    # We iterate through both pose and arduino data lists. 
    for i, j in zip(pose_raw, arduino_data):

        # Convert the string representation of a list into an actual list
        i_list = ast.literal_eval(i)

        pose.append(i_list)
        # Check if arduino data is NOT an empty entry
        if isinstance(j, str) and j.strip():
            # If not, then append the stimulation information
            # print(file)
            try:
                # direction = j[0]
                # number = j[1:]
                # if direction == 'E': 
                #     direction = 'Both'
                # elif direction == 'A': 
                #     direction = 'Left'
                # elif direction == 'B': 
                #     direction ='Right'
                direction, number = j.split(", ")
                freq = int(number[:2])
                freq = int(number)
                stim_deets.append([direction, freq])
                stim_occur.append(1)
            except ValueError: 
                # If empty, no stimulation: [None, None, None]
                stim_deets.append([None, None])
                stim_occur.append(0)
        else: 
            # If empty, no stimulation: [None, None, None]
            stim_deets.append([None, None])
            stim_occur.append(0)


    # Return relevant lists. 
    return pose, stim_deets, stim_occur, fps



def exp_weighted_ma(part, alpha):
    """An application of an exponential weighted moving average filter to 
    smooth data - alpha close to 1 means minimal smoothing"""
    # Initialize empty lists to store x and y coordinates separately
    partx = []
    party = []
    
    # Separate x and y coordinates from the input list
    for i in part: 
        partx.append(i[0])
        party.append(i[1])
    
    # Convert lists to pandas Series for easier manipulation
    partx = pd.Series(partx)
    party = pd.Series(party)
    
    # Apply exponential weighted moving average to x coordinates
    # Round to 5 decimal places and convert back to list
    partx = round(partx.ewm(alpha, adjust=False).mean(), 5)
    partx = partx.tolist()

    # Apply exponential weighted moving average to y coordinates
    # Round to 5 decimal places and convert back to list
    party = round(party.ewm(alpha, adjust=False).mean(), 5)
    party = party.tolist()

    # Combine smoothed x and y coordinates back into a single list
    smooth_data = []
    for i in range(len(partx)):
        smooth_data.append([partx[i], party[i]])

    # Return the smoothed data
    return smooth_data

def remove_outliers_and_smooth(data, alpha=0.1, z_thresh=2):
    """
    Removes outliers from 2D data and applies EWMA smoothing.
    - data: list of [x, y] points
    - alpha: EWMA smoothing factor (0 < alpha <= 1)
    - z_thresh: z-score threshold for outlier detection
    """
    # Convert to numpy array for easier math
    arr = np.array(data)
    x, y = arr[:, 0], arr[:, 1]
    
    # Outlier detection using z-score
    z_x = np.abs(stats.zscore(x, nan_policy='omit'))
    z_y = np.abs(stats.zscore(y, nan_policy='omit'))
    mask = (z_x < z_thresh) & (z_y < z_thresh)
    
    # Remove outliers
    # x_clean, y_clean = x[mask], y[mask]
    
    # Or maybe we can replace outliers with NaN and interpolate 
    x_clean = x.copy()
    y_clean = y.copy()
    x_clean[~mask] = np.nan
    y_clean[~mask] = np.nan
    x_clean = pd.Series(x_clean).interpolate().bfill().ffill().values
    y_clean = pd.Series(y_clean).interpolate().bfill().ffill().values
    
    # Apply EWMA smoothing
    x_smooth = pd.Series(x_clean).ewm(alpha=alpha, adjust=False).mean().values
    y_smooth = pd.Series(y_clean).ewm(alpha=alpha, adjust=False).mean().values
    
    # Combine back to [x, y] pairs
    smoothed_data = np.column_stack((x_smooth, y_smooth)).tolist()
    return smoothed_data # mask shows which points were kept


def remove_outliers_and_smooth_1d(data, alpha=0.1, z_thresh=2):
    """
    Removes outliers from 1D data and applies EWMA smoothing.
    - data: list of numeric values
    - alpha: EWMA smoothing factor (0 < alpha <= 1)
    - z_thresh: z-score threshold for outlier detection
    """
    arr = np.array(data, dtype=float)  # convert to float for NaN support

    # Outlier detection using z-score
    z = np.abs(stats.zscore(arr, nan_policy='omit'))
    mask = z < z_thresh

    # Replace outliers with NaN and interpolate
    clean = arr.copy()
    clean[~mask] = np.nan
    clean = pd.Series(clean).interpolate().bfill().ffill().values

    # Apply EWMA smoothing
    smooth = pd.Series(clean).ewm(alpha=alpha, adjust=False).mean().values

    return smooth.tolist()


def body_vel(middle, bottom, fps):
    """Calculate the in-line and transverse velocities of the beetle.
    
    Args:
        middle (list): positional data for the middle point of beetle
        bottom (list): positional data for the bottom point of beetle
        fps (int): frames per second that the data has been recorded at. 
    
    Returns:
        tuple: A tuple containing lists of in-line velocity and signed transverse velocity.
               Transverse velocity is negative in one direction and positive in the opposite direction.
    """

    body_v_in_line = []
    body_v_transverse = []

    # Define vertical axis for determining lateral direction (assuming 2D plane)
    vertical_axis = np.array([0, 1])

    for i in range(3, len(middle)):
        # Velocity vector of middle point
        delta = np.subtract(middle[i], middle[i-2])

        # Body axis (direction from bottom to middle)
        body_axis = np.subtract(middle[i], bottom[i])
        body_axis_norm = np.linalg.norm(body_axis)

        # Normalize body axis
        if body_axis_norm := np.linalg.norm(body_axis):
            body_axis_unit = body_axis / body_axis_norm
        else:
            body_axis_unit = np.zeros_like(body_axis)

        # Calculate perpendicular vector to body axis (rotated 90 degrees CCW)
        perp_body_axis_unit = np.array([-body_axis_unit[1], body_axis_unit[0]])

        # Calculate velocities
        in_line_velocity = np.dot(delta, body_axis_unit)
        transverse_velocity = np.dot(delta, perp_body_axis_unit)

        # Scale velocities
        scale_factor = fps / body_axis_norm if body_axis_norm != 0 else 0
        body_v_in_line.append(in_line_velocity * scale_factor)
        body_v_transverse.append(transverse_velocity * scale_factor)

        # Lateral velocity with sign using dot product with perpendicular axis
        lateral_velocity_signed = np.dot(delta, perp_body_axis_unit) * scale_factor
        body_v_transverse.append(lateral_velocity_signed)

    # Exponential smoothing
    alpha = 0.5
    body_v_in_line = pd.Series(body_v_in_line).ewm(alpha=alpha, adjust=False).mean()
    body_v_in_line = round(body_v_in_line, 5).tolist()
    
    body_v_transverse = pd.Series(body_v_transverse).ewm(alpha=alpha, adjust=False).mean()
    body_v_transverse = round(body_v_transverse, 5).tolist()
    
    # Normalization (baseline subtraction)
    ref_idx = int(0.1 * fps)
    if ref_idx >= len(body_v_in_line):
        ref_idx = 0

    ref_in_line = body_v_in_line[ref_idx]
    ref_transverse = body_v_transverse[ref_idx]

    body_v_in_line = [v - ref_in_line for v in body_v_in_line]
    body_v_transverse = [v - ref_transverse for v in body_v_transverse]

    return body_v_in_line, body_v_transverse



def get_body_angles(angles, fps):
    # Initialize normalized angles list with the first angle
    
    normalized_angles = [angles[0]] 
    
    # Normalize subsequent angles to avoid large jumps
    for i in range(1, len(angles)):
        # Calculate the difference between current and previous angle
        delta = angles[i] - angles[i - 1]
        
        # Adjust for jumps greater than 180 degrees
        # This ensures the smallest angle difference is always used
        delta = (delta + 180) % 360 - 180
        
        # Add the adjusted delta to the previous normalized angle
        normalized_angles.append(normalized_angles[-1] + delta)

    reference = reference = normalized_angles[int(0.15 * fps)]
    # Return the list of normalized angles
    return [angle - reference for angle in normalized_angles]


def get_ang_vel(angles, fps):
    """Calculate angular velocity (degs/s) from a list of angles over uniform time intervals (specify fps)."""
    if len(angles) < 2:
        return []  # Not enough data to calculate velocity

    angular_velocities = []
    time_interval = 1 / fps  # Time interval between measurements in seconds (100fps)

    for i in range(2, len(angles)):
        delta_angle = angles[i] - angles[i - 1]  # Change in angle
        angular_velocity = delta_angle / time_interval  # Angular velocity = delta_angle / delta_time
        angular_velocities.append(angular_velocity)

    return angular_velocities



def get_post_stim(pose, stim_deets, stim_occur, fps):
    """
    Extracts data occurring just before and after a stimulation.
    
    This function should be used BEFORE applying EWMA filters to extract moments of interest.
    The post_frames and pre_frames variables can be adjusted to change the extraction window.
    
    Args:
    pose (list): List containing x, y, angle details.
    stim_deets (list): List containing stimulation details.
    stim_occur (list): List indicating stimulation occurrences (1 for stimulation, 0 otherwise).
    fps (int): Frames per second of the recording.

    Returns:
    tuple: A tuple containing stimulation details and extracted body part coordinates.
    """
    #Define the dictionary
    stim_dict = {}

    # Define the extraction window
    post_frames = int(fps * 1.0)  # 0.7 seconds after stimulation
    pre_frames = int(fps * 0.15)   # 0.3 seconds before stimulation
    
    # Find indices where stimulation occurred
    stim_index = [i for i, x in enumerate(stim_occur) if x == 1]
    
    
    # Extract data for the last stimulation
    for stim in stim_index:
        start = stim - pre_frames
        end = stim + post_frames
        if start < 0 or end > len(pose): 
            continue
        # Extract body part coordinates within the defined window
        pose_sect = pose[stim-pre_frames:stim+post_frames]
        
        
        # Get stimulation details
        side = stim_deets[stim][0]
        freq = stim_deets[stim][1]

        if (side, freq) not in stim_dict: 
            stim_dict[(side, freq)] = []

        stim_dict[(side, freq)].append(pose_sect)

    return stim_dict


def statistical_significance(data1, data2): 
    # Convert data to numpy arrays
    array1 = np.array(data1)
    array2 = np.array(data2)

    # Perform independent samples t-test
    t_statistic, p_value = stats.ttest_ind(array1, array2)

    # Print results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    # Interpret the results
    alpha = 0.05  # Set your significance level
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between the two groups.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the two groups.")


            
def success_rate(forward_velocity, body_angles, vel_thresh=0.2, angle_thresh=2):
    """
    Args:
        forward_velocity, body_angles: dicts of lists (trials) per key
        vel_thresh: threshold for velocity increase (units)
        angle_thresh: threshold for angle change (degrees)
    Returns:
        fwd_vel_success, body_angle_success: lists of 1 (success) or 0 (fail) for each trial
    """
    fwd_vel_success = []
    body_angle_success = []

    # Helper function to compute baseline and during-stim medians
    def get_medians(trace):
        n = len(trace)
        pre_stim = trace[:int(0.15/1.15*n)]
        during_stim = trace[int(0.15/1.15*n):int(0.65/1.15*n)]
        return abs(np.mean(pre_stim)), abs(np.mean(during_stim))

    # Forward velocity
    for key, trials in forward_velocity.items():
        for trial in trials:
            base, stim = get_medians(trial)
            if stim - base >= vel_thresh:
                fwd_vel_success.append(1)
            else:
                fwd_vel_success.append(0)

    # Body angle
    for key, trials in body_angles.items():
        for trial in trials:
            base, stim = get_medians(trial)
            if abs(stim - base) >= angle_thresh:
                body_angle_success.append(1)
            else:
                body_angle_success.append(0)

    return fwd_vel_success, body_angle_success


def angle_interpolate(values): 
    angles_deg = [math.degrees(x) if x is not None else float('nan') for x in values]
    arr = np.array(angles_deg, dtype = np.float64)
    nans = np.isnan(arr)
    if nans.any(): 
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])

    return arr.tolist()


def pos_interpolate(pos):
    """
    pos: list of [x, y] (with values or None)
    Returns a list of [x, y] with None values interpolated.
    """
    pos_array = np.array([
        [v if v is not None else np.nan for v in pt]
        for pt in pos
    ], dtype=np.float64)  # shape (n, 2)

    # Interpolate x and y independently
    for i in range(2):  # For x and y
        col = pos_array[:, i]
        nans = np.isnan(col)
        if nans.any() and (~nans).any():
            col[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(~nans),
                col[~nans]
            )
        pos_array[:, i] = col

    # Convert back to list of lists
    return pos_array.tolist()


def trial_is_outlier(angles, key): 
    side = key[0]
    if side == "Right": 
        for angle in angles: 
            if angle > 30: 
                return True
    elif side == "Left": 
        for angle in angles: 
            if angle < -30: 
                return True 
    else: 
        return False
    
