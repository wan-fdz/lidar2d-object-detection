""" LidarDataObjectDetection.py

    Authors:
        Vanessa Alejandra Fernandez Vega
    Affiliation: Tecnoap
    Contact:
        vfernandez@tecnoap.com
    First created: Ago-14-2022
    Last updated: Ago-15-2022
    This code has the object detection functions for the main program "LidarDataVisualization.py". 
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def cluster_df(df, eps=0.225, min_samples=6):
    """
    This function finds the clusters in the dataFrame sent.

    :param
        df: dataFrame
        eps: the distance to neighbors in a cluster 
        min_samples: the minimum number of samples required to form a cluster
    :return:
        labels: array with the cluster's labels
    """    
    # Clusterization Utilizing DBSCAN with eps=0.225, min_points=6
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    labels = clusters.labels_

    return labels

def visualization_clustered_df(labels, df):
    """
    This function visualizes the clusters found in the DataFrame sent.

    :param
        df: dataFrame
        labels: array with the cluster's labels
    :return:
        None
    """ 
    p = sns.scatterplot(data=df,  x="x", y="y", hue=labels, legend="full", palette="deep")
    sns.move_legend(p, "upper right", title='Clusters')
    plt.show(block=True)

    return None

def clean_dataframe(dataframe):
    """
    This function deletes the labels indicated with "-1".
    These labels are considered "noise" in the DataFrame.

    :param
        dataframe: point cloud dataframe
    :return:
        cleaned_dataframe: point cloud dataframe without noise
    """
    #Drop all the labels with "-1"
    cleaned_dataframe = dataframe[dataframe.Label != -1]

    return cleaned_dataframe


def export_clusters(dataframe, labels):
    """
    This function reads all the clusters found per scan and exports 
    the dynamic cluster's point cloud dataframe along with it's label.

    :param
        dataframe: point cloud dataframe
        labels: array with the cluster's labels
    :return:
        cluster_of_interest: dynamic cluster's point cloud dataframe
    """ 

    #Declare the dynamic cluster
    cluster_of_interest = pd.DataFrame(columns=['x', 'y', 'z', 'Label'])

    #Iteration of all the clusters per scan
    for i in range(labels.max()):

        #Declare the local dynamic cluster
        cluster = pd.DataFrame(columns=['x', 'y', 'z', 'Label'])

        #Insert the point cloud data in the local dynamic cluster
        cluster.x = dataframe.x[dataframe['Label'] == i]
        cluster.y = dataframe.y[dataframe['Label'] == i]
        cluster.z = dataframe.z[dataframe['Label'] == i]
        cluster['Label'] = i

        #Compute the width of the cluster
        width = (cluster.x.max() - cluster.x.min())/1000.0
        height = (cluster.y.max() - cluster.y.min())/1000.0

        #By knowing beforehand an estimated width of the dynamic object along
        # with an estimated amount of points, the cluster's width and 
        # dataframe's length are compared to a certain range 

        if 150 <= len(cluster) <= 170 and 1.0 <= width <= 5.0:

            #Insert the local dynamic cluster into the main one
            cluster_of_interest = cluster
            cluster_labels = np.full(len(cluster), i)

        
            #visualization_clustered_df(cluster_labels, cluster_of_interest)

        #Delete the local dynamic cluster
        del cluster
    return cluster_of_interest


def compute_centroid(dataframe):
    """
    This function computes the centroid of each dynamic cluster found.

    :param
        dataframe: point cloud dataframe
    :return:
        x_center: x coordinate of the cluster's centroid
        y_center: y coordinate of the cluster's centroid
        z_center: z coordinate of the cluster's centroid
    """ 
    #Initialize the coordinates
    x_center = -1
    y_center = -1
    z_center = -1

    #Compute the centroid calculation
    if not dataframe.empty:
        x_center = dataframe.x.sum()/len(dataframe)
        y_center = dataframe.y.sum()/len(dataframe)
        z_center = dataframe.z.sum()/len(dataframe)

    return x_center, y_center, z_center