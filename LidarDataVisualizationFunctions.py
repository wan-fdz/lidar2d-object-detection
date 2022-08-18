""" LidarDataVisualizationFunctions.py

    Authors:
        Vanessa Alejandra Fernandez Vega
    Affiliation: Tecnoap
    Contact:
        vfernandez@tecnoap.com
    First created: Ago-14-2022
    Last updated: Ago-15-2022
    This code has the functions for the main program "LidarDataVisualization.py". 
"""

# Import standard libraries
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time


def read_lidar_data(file_name):
    """
    This function opens a txt dataFile provided by the user
    and reads it line by line while appending every sentence
    into a python list.

    :param 
        file_name: dataset file name
    :return: 
        timestamps: list of serial timestamps
        angles: list of azimuth scanning angles
        ranges: list of ranges
    """

    # Open the file and read each line
    with open(file_name) as fp:
        line = fp.readline()
        strips = []
        cnt = 1

        # Reads lidar data and stores it as a list in "strips"
        while line:
            currentStrip = line.strip()
            strips.append(currentStrip)
            line = fp.readline()
            cnt += 1

    timestamps, angles, ranges = txt_data_conversion(strips)

    return timestamps, angles, ranges

def txt_data_conversion(strips):
    """
    This functions converts the lidar data lists into new ones without the invalid
    characters and separated by commas.

    :param 
        strips: list with the lidar data read
    :return: 
        timestamps_df_list: list of serial timestamps 
        angles_df_list: list of azimuth scanning angles
        ranges_df_list: list of ranges
    """

    # Sentences with [][][] shape
    new_strips = []

    for i in range(len(strips)):
        string = str(strips[i])
        new_strips.append(re.findall(r'\[.*?\]', string))

    timestamps = []
    angles = []
    ranges = []

    # Delete invalid characters (brackets)
    for i in range(len(new_strips)):
        current_timestamp = new_strips[i][0]
        new_timestamp = current_timestamp.replace('[', '')
        new_timestamp = new_timestamp.replace(' ', '')
        new_timestamp = new_timestamp.replace(']', '')
        timestamps.append(new_timestamp)

        current_angle = new_strips[i][1]
        new_angle = current_angle.replace('[', '')
        new_angle = new_angle.replace(' ', '')
        new_angle = new_angle.replace(']', '')
        angles.append(new_angle)

        current_range = new_strips[i][2]
        new_range = current_range.replace('[', '')
        new_range = new_range.replace(' ', '')
        new_range = new_range.replace(']', '')
        ranges.append(new_range)

    # Separate by commas
    for i in range(len(timestamps)):
        angles[i] = angles[i].split(',')
        ranges[i] = ranges[i].split(',')


    lidar_df = pd.DataFrame(columns=['timestamps', 'angles', 'ranges'])
    lidar_df.timestamps = timestamps
    lidar_df.angles = angles
    lidar_df.ranges = ranges

    # Convert from str to float
    timestamps_df_list = [float(sublist) for sublist in timestamps]
    angles_df_list = [[float(x) for x in sublist] for sublist in angles]
    ranges_df_list = [[float(x) for x in sublist] for sublist in ranges]

    return timestamps_df_list, angles_df_list, ranges_df_list


def polar_to_cartesian(angles, ranges):
    """
    This function converts the points given from the lidar data from
    Polar to Cartesian coordinate frame.
    :param 
        angles: list with azimuth scanning angles
        ranges: list with ranges
    :return: 
        df: dataframe with the cartesian coordinates
    """
    # Use one scan only to make the conversion
    np_angles = np.array(angles)
    np_ranges = np.array(ranges)

    # Point conversion
    ox = np.cos(np_angles) * np_ranges
    oy = np.sin(np_angles) * np_ranges

    # Save the coordinates in a new dataframe
    df = pd.DataFrame(columns=['x', 'y', 'z'])
    df.x = ox
    df.y = oy
    z = np.zeros_like(ox)
    df.z = z

    return df

def lidar_2d_mapping(original_df):
    """
    This function makes the lidar 2D mapping.
    :param 
        original_df: dataframe with the cartesian coordinates
    :return: 
        Nothing to return.
    """

    """
    # Debug lines
    # Print the original dataframe 
    print("Original dataframe")
    print(original_df)
    """

    # Insert a figure
    fig = go.Figure()

    # Add the traces of both dataframes
    fig.add_trace(go.Scatter(x=original_df.x, y=original_df.y, mode='markers', name='Original dataframe'))
    
    # Show the figure
    fig.show()

    return None

def visualisation_360_deg_scans(angles, ranges, clustered_df):
    """
    This function plots the 360 degree scans recorded by the lidar in polar and cartesian coordinates.
    :param 
        angles: list of azimuth scanning angles
        ranges: list of ranges
    :return: 
        Nothing to return.
    """

    #Convert the points given from each scan from Polar to Cartesian coordinate frame.

    df = []
    df_og = []
    
    for i in range(len(angles)):
       df.append(polar_to_cartesian(angles[i],ranges[i]))

    for i in range(len(angles)):
        np_angles = np.array(angles[i])
        np_ranges = np.array(ranges[i])

        # Save the coordinates in a new dataframe
        df_polar = pd.DataFrame(columns=['theta', 'r'])
        df_polar.theta = np_angles
        df_polar.r = np_ranges
        df_og.append(df_polar)

    # Plot the first scan
    x = df[0].x
    y = df[0].y

    theta = df_og[0].theta
    r = df_og[0].r

    x_clustered = clustered_df[0].x
    y_clustered = clustered_df[0].y
    clean_labels_list = clustered_df[0].color.to_numpy().tolist()

    plt.ion()

    gs = gridspec.GridSpec(2, 2)

    # Create two subplots and unpack the output array immediately
    figure = plt.figure()
    ax1 = figure.add_subplot(gs[0,0])
    line1, = ax1.plot(x, y, 'bo')
    ax1.set_title('Cartesian')

    ax2 = figure.add_subplot(gs[0,1], projection='polar')
    line2, = ax2.plot(theta, r, 'bo')
    ax2.set_title('Polar')
    
    ax3 = figure.add_subplot(gs[1,:])
    line3 = ax3.scatter(x_clustered, y_clustered, c=clean_labels_list)
    ax3.set_title('Clustered and Noise Free data')

    # Update the data from each scan 
    for p in range(34):
        updated_x = df[p].x
        updated_y = df[p].y

        updated_theta = df_og[p].theta
        updated_r = df_og[p].r

        clustered_df_aux = pd.DataFrame(clustered_df[p])
        updated_xy = clustered_df_aux[['x', 'y']].to_numpy()
        clean_labels_list = clustered_df[0].Label.to_numpy().tolist()
          
        line1.set_xdata(updated_x)
        line1.set_ydata(updated_y)

        line2.set_xdata(updated_theta)
        line2.set_ydata(updated_r)

        line3.set_offsets(updated_xy)
        
        figure.canvas.draw()
        
        figure.canvas.flush_events()
        time.sleep(0.1)

    return None