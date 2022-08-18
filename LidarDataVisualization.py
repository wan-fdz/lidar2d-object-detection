"""LidarDataVisualization.py
    Authors:
        Vanessa Alejandra Fernandez Vega
    Affiliation: Tecnoap
    Contact:
        vfernandez@tecnoap.com
    First created: Ago-14-2022
    Last updated: Ago-15-2022
"""

# Import user libraries
import LidarDataVisualizationFunctions as lidar_functions
import LidarDataObjectDetection as object_detection
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import statistics
import numpy as np
import pandas as pd

lidar_file_name = 'sick_dataset.txt'

# Driver function
if __name__ == "__main__":

    # Read the lidar dataset and return the timestamps, angles and ranges
    timestamps, angles, ranges = lidar_functions.read_lidar_data(file_name=lidar_file_name)

    # Convert the points from Polar to Cartesian coordinate frame.
    df = lidar_functions.polar_to_cartesian(angles=angles[0], ranges=ranges[0])

    # 2D mapping of the lidar data (Original and Transformed)
    #lidar_functions.lidar_2d_mapping(original_df=df)

    # Start the object detection and localization

    df_scans = []
    nf_df = []

    for i in range(len(angles)):
       df_scans.append(lidar_functions.polar_to_cartesian(angles[i],ranges[i]))

    # Object of interest's centroids dataframe creation
    coordinates = pd.DataFrame(columns=["X", "Y", "Z"])
    j = 0
    estimated_speed = []
    first_time = True
    cont = 0

    # # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
    # nbrs = NearestNeighbors(n_neighbors=5).fit(df_scans[0])
    # # Find the k-neighbors of a point
    # neigh_dist, neigh_ind = nbrs.kneighbors(df_scans[0])
    # # sort the neighbor distances (lengths to points) in ascending order
    # # axis = 0 represents sort along first axis i.e. sort along row
    # sort_neigh_dist = np.sort(neigh_dist, axis=0)

    # k_dist = sort_neigh_dist[:, 4]
    # plt.plot(k_dist)
    # plt.axhline(y=2.5, linewidth=1, linestyle='dashed', color='k')
    # plt.ylabel("k-NN distance")
    # plt.xlabel("Sorted observations (4th NN)")
    # plt.show()


    # Iteration of the lidar scans
    for i in range(len(df_scans)):

        # Clusterization Utilizing DBSCAN with eps=0.225, min_points=15
        labels = object_detection.cluster_df(df=df_scans[i], eps=120, min_samples=6)

        if len(labels) != 0:

            # Debugging lines to visualize the clusters per scan
            #object_detection.visualization_clustered_df(labels, df=df_scans[i])

            # Add label to dataframe
            df_clustered = df_scans[i]
            df_clustered["Label"] = labels

            # Noise cleaning
            noise_free_df  = object_detection.clean_dataframe(df_clustered).copy(deep=True)
            del df_clustered
            colors = {0:'#7400B8', 1:'#6930C3', 2:'#5E60CE', 3:'#5390D9', 4:'#4EA8DE', 5:'#48BFE3', 6:'#56CFE1', 7:'#64DFDF', 8:'#72EFDD', 9:'#80FFDB', 10:'#81DF20'}
            noise_free_df["color"] = noise_free_df["Label"].map(colors)
            nf_df.append(noise_free_df)

            # Quitar labels y color
            # noise_free_df = noise_free_df.drop('color', axis=1)
            
            # Extraction of clusters of interest
            cluster_of_interest = object_detection.export_clusters(noise_free_df, labels)

            # Computation of centroids
            xc, yc, zc = object_detection.compute_centroid(cluster_of_interest)

            # Centroids append to Dataframe
            if xc != -1 and yc != -1 and zc != -1:
                coordinates.loc[j, "X"] = xc
                coordinates.loc[j, "Y"] = yc
                coordinates.loc[j, "Z"] = zc
                j += 1

                if first_time:
                    estimated_speed.append(0.0)
                    yc_ant = yc
                    first_time = False
                    cont+=1
                else:
                    speed = (yc - yc_ant) / (0.00000916*1000000.0)
                    estimated_speed.append(speed)
                    yc_ant = yc
                    cont+=1
            
    print("The estimated speed truck is: ", abs(statistics.mean(estimated_speed)), " km/hr")

    #Visualise the scans in Polar and X-Y coordinates
    lidar_functions.visualisation_360_deg_scans(angles=angles, ranges=ranges, clustered_df=nf_df)
