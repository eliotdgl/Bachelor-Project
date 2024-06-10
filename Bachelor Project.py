# Dugelay Eliot - Bachelor Project

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd
import shapely.geometry
import geopy.distance
from datetime import datetime
from windrose import WindroseAxes
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm


# Initial global configuration for matplotlib
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



# ------- FUNCTIONS ------- #


def clean_data(data):
    """
    Clean the data.
    
    Parameters:
        data (Pandas DataFrame).
        
    Returns:
        clean_d (Pandas DataFrame): Cleaned data.
    """
    
    # Keep only specific columns
    columns = ['Site Name','Report Date','Time Period Ending','Avg mph','Total Volume']
    clean_d = data[columns].copy()
    

    # Change 'Report Date' format
    clean_d['Report Date'] = clean_d['Report Date'].str.replace('00:00:00','')
    clean_d['Report Date'] = pd.to_datetime(data['Report Date'], format='%d/%m/%Y %H:%M:%S')
    
    
    # Replace NA 'Avg speed' values by an average on the data
    clean_d['Avg mph'].fillna(clean_d['Avg mph'].mean(), inplace=True)
    
    # Convert average speed in km/h
    clean_d['Avg mph'] *= 1.609344
    clean_d.rename(columns={'Avg mph': 'Avg speed (km/h)'}, inplace=True)
            
            
    return clean_d


def interpolate_na(data):
    """
    Replace NA 'Total Volume' values by an average on similar data (same time)
    
    Parameters:
        data (Pandas DataFrame).
            
    Returns:
        data (Pandas DataFrame): Without NA values
    """
    
    data['Time Period Ending'] = pd.to_datetime(data['Time Period Ending'], format='%H:%M:%S')

    for index, row in data.iterrows():
        if pd.isna(row['Total Volume']):
            
            start_time = pd.to_datetime(row['Time Period Ending'] - pd.Timedelta(minutes=3))
            end_time = pd.to_datetime(row['Time Period Ending'] + pd.Timedelta(minutes=3))

            data_similar = data[(data['Site Name']==row['Site Name']) &
                                   (data['Time Period Ending'] >= start_time) &
                                   (data['Time Period Ending'] <= end_time) &
                                   (data['Total Volume'].notna())]
            
            data.at[index,'Total Volume'] = data_similar['Total Volume'].mean()
    
    return data


def add_features(data):
    """
    Add features to the data: Flow (veh/h) and Density (veh/meter).
    
    Parameters:
        data (Pandas DataFrame).
        
    Returns:
        distances_dict.
    """
    
    # Compute the flow Q using 'Total Volume' and the 15 minutes intervals
    data['Flow Q (veh/h)'] = data['Total Volume'] / 0.25  # 15min = 0.25h
    
    # Compute the density rho using Flow and Average speed (with convertion of km/h into meter per hour)
    data['Rho (veh/meter)'] = data['Flow Q (veh/h)'] / (data['Avg speed (km/h)']*1000)



def availability(df, title):
    """
    Check the data availability.
    
    Parameters:
        df (Pandas DataFrame): data.
        title (string): Name of the DataFrame.
    """
    
    df_nona = df.dropna(subset=['Total Volume'])
    
    #Determine the number of scans for each scanner each day
    availability_per_day = df_nona.groupby(['Site Name', df['Report Date'].dt.date]).size().unstack(level=0).fillna(0)
    
    #Convert in percentage (15mins time intervals: (24h*60mins)/15mins = 96 scans)
    availability_per_day = availability_per_day/96*100
    
    #Plots
    plt.figure()
    availability_per_day.plot(marker='o', linestyle='-', ax=plt.gca())
    plt.title('Availability for $\mathbf{'+title+'}$ each day of January for each scanner')
    plt.xlabel('Date')
    plt.ylabel('Availability (%)')
    plt.ylim([0,105])
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Scanner')
    plt.show()


def traffic_flow(df, title, subtitle='', start_date='2023-01-01', end_date='2023-01-07', ax=False, descriptions=True):
    """
    Plot the traffic flow for each scanner according to time.
    
    Parameters:
        df (Pandas DataFrame): data.
        title (string): Name of the DataFrame.
        subtitle (string).
        number_days (int).
        start_date, end_date (string).
        ax (matplotlib plot).
        descriptions (bool).
    """
    
    # Check if end_date is after (or ==) start_date
    assert((datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days >= 0), 'When calling traffic_flow(), end_date cannot be before start_date.'
    
    number_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
    
    # Create figure if not already given when calling the function
    if not(ax):
        fig, ax = plt.subplots(1, 1, figsize=(12,8), tight_layout=True)
       
    # Number of scans in case of complete data
    number_intervals_per_day = 96
    total_intervals = number_intervals_per_day * number_days
    number_scans = np.arange(1, total_intervals + 1)
    
    title_end =''
    
    # Keep data only for the wanted week
    week_data = df[(df['Report Date'] >= start_date) & (df['Report Date'] <= end_date)]
    
    scanners = week_data['Site Name'].unique()
    
    for scanner in scanners:
        # Keep data only for the selected scanner
        scanner_data = week_data[week_data['Site Name'] == scanner]
        
        # Check if the data are complete: if not, do not plot it and indicate 'data incomplete for scanner ...'
        if len(scanner_data['Total Volume']) == total_intervals:
            ax.plot(number_scans, scanner_data['Total Volume'], label=scanner)
        else:
            title_end += ' (Data incomplete for '+scanner+')'
            print(f'Some data are missing for scanner {scanner}')
    
    #Plot
    if not(ax) or descriptions:
        ax.set_title('Traffic Flow for $\mathbf{'+title+'}$ scanned by each scanner every 15 minutes\nbetween '+start_date+' and '+end_date+'\n'+title_end)
        ax.set_ylabel('Number of cars')
        ax.legend(title='Scanner', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.set_title(subtitle)
    
    ax.grid()
    
    # Redefine x-axis
    if number_days == 1: # For a day
        ax.set_xticks(np.arange(1, total_intervals + 1, number_intervals_per_day // 24))
        ax.set_xticklabels(range(24), rotation=45)
    else:
        ax.set_xticks(np.arange(1, total_intervals + 1, number_intervals_per_day))
        date_range = pd.date_range(start=start_date, periods=number_days, freq='D')
        ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in date_range], rotation=45)


def avg_speed(df, title, subtitle='', start_date='2023-01-01', end_date='2023-01-07', ax=False, descriptions=True):
    """
    Plot the average speed for each scanner according to time.
    
    Parameters:
        df (Pandas DataFrame): data.
        title (string): Name of the DataFrame.
        subtitle (string).
        number_days (int).
        start_date, end_date (string).
        ax (matplotlib plot).
        descriptions (bool).
    """
    
    # Check if end_date is after (or ==) start_date
    assert((datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days >= 0), 'When calling traffic_flow(), end_date cannot be before start_date.'
    
    number_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
    
    # Create figure if not already given when calling the function
    if not(ax):
        fig, ax = plt.subplots(1, 1, figsize=(12,8), tight_layout=True)
       
    # Number of scans in case of complete data
    number_intervals_per_day = 96
    total_intervals = number_intervals_per_day * number_days
    number_scans = np.arange(1, total_intervals + 1)
    
    title_end =''
    
    # Keep data only for the wanted week
    week_data = df[(df['Report Date'] >= start_date) & (df['Report Date'] <= end_date)]
    
    scanners = week_data['Site Name'].unique()
    
    for scanner in scanners:
        # Keep data only for the selected scanner
        scanner_data = week_data[week_data['Site Name'] == scanner]
        
        # Check if the data are complete: if not, do not plot it and indicate 'data incomplete for scanner ...'
        if len(scanner_data['Avg speed (km/h)']) == total_intervals:
            ax.plot(number_scans, scanner_data['Avg speed (km/h)'], label=scanner)
        else:
            title_end += ' (Data incomplete for '+scanner+')'
            print(f'Some data are missing for scanner {scanner}')
    
    #Plot
    if not(ax) or descriptions:
        ax.set_title('Average Speed for $\mathbf{'+title+'}$ scanned by each scanner every 15 minutes\nbetween '+start_date+' and '+end_date+'\n'+title_end)
        ax.set_ylabel('Average speed (km/h)')
        ax.legend(title='Scanner', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.set_title(subtitle)
    
    ax.grid()
    
    # Redefine x-axis
    if number_days == 1: # For a day
        ax.set_xticks(np.arange(1, total_intervals + 1, number_intervals_per_day // 24))
        ax.set_xticklabels(range(24), rotation=45)
    else:
        ax.set_xticks(np.arange(1, total_intervals + 1, number_intervals_per_day))
        date_range = pd.date_range(start=start_date, periods=number_days, freq='D')
        ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in date_range], rotation=45)


def plot_wind(wind_data, source_name):
    """
    Plot the wind data.
    
    Parameters:
        wind_data (Pandas DataFrame).
        source_name (string): Name of the source which the data are from.
    """
    
    # Plot the Wind Speed according to the date
    plt.figure(figsize=(10, 6))
    plt.plot(wind_data['date'], wind_data['windspeed'])
    plt.title('Wind speed')
    plt.xlabel('Date')
    plt.ylabel('Wind Speed (km/h)')
    plt.text(wind_data['date'].iloc[-1], wind_data['windspeed'].max()-1, f'Source: {source_name}',
         ha='right', va='bottom', fontsize=9, color='black') # Add source on the plot
    plt.grid()
    plt.show()

    # Create wind rose plot for Wind Direction
    plt.figure(figsize=(10, 8))
    ax = WindroseAxes.from_ax()
    ax.bar(wind_data['winddir'], wind_data['windspeed'], normed=True, opening=0.8, edgecolor='white')
    ax.set_legend(title='Wind speed (km/h)')
    plt.title('Wind rose')
    plt.text(1, -0.15, f'Source: {source_name}', ha='right', va='bottom', fontsize=9, transform=plt.gca().transAxes) # Add source on the plot
    plt.show()


def plots_wind_pol_traffic(df_traffic, df_wind, df_pollution, start_date, end_date):
    """
    Plot wind speed, pollution and traffic flow for comparison.
    
    Parameters:
        df_traffic (Pandas DataFrame)
        df_wind (Pandas DataFrame).
        df_pollution (Pandas DataFrame).
        start_date, end_date (string).
    """
    
    df_wind_week = df_wind[(df_wind['date'] >= start_date) & (df_wind['date'] <= end_date)]
    df_pollution_week = df_pollution[(df_pollution['End Date'] >= start_date) & (df_pollution['End Date'] <= end_date)]
    
    fig, axes = plt.subplots(nrows=3, ncols=1)
    
    # Wind Speed Plot
    axes[0].plot(df_wind_week['date'], df_wind_week['windspeed'], color='blue')
    axes[0].set_title('Wind Speed')
    axes[0].axes.get_xaxis().set_visible(False)
    axes[0].axes.get_yaxis().set_visible(False)
    
    # Pollution Concentration Plot
    axes[1].plot(df_pollution_week['NO2'], color='green')
    axes[1].set_title('NO$_2$ Concentration')
    axes[1].axes.get_xaxis().set_visible(False)
    axes[1].axes.get_yaxis().set_visible(False)
    
    # Traffic Plot
    traffic_flow(df_c, 'M25-Clockwise', start_date='2023-01-16', end_date='2023-01-22', ax=axes[2], descriptions=False)
    axes[2].set_title('Traffic Flow')
    axes[2].axes.get_yaxis().set_visible(False)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth using their longitude and latitude.
    
    Parameters:
        lat1, lon1 (float): Latitude and Longitude of point 1 (in degrees).
        lat2, lon2 (float): Latitude and Longitude of point 2 (in degrees).
        
    Returns:
        distance (float): Distance between the two points in meters.
    """
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    radius_earth = 6371  # Radius of the Earth in kilometers
    distance = radius_earth * c * 1000  # Convert to meters
    
    return distance


def shape_grid_graph(gdf, scanner_coords, pos_coords):
    """
    Plot roads map, Create and plot grid cells and graph.
    
    Parameters:
        gdf (Geopandas): Roads map.
        scanners_coords (Dictionary): Coordinates of the scanners.
        pos_coords (Dictionary): Coordinates of the nodes.
        
    Returns:
        G (NetworkX graph): Graph of the roads network.
        distance_matrix (numpy array): Distances matrix A of size (num_cells, num_cells, num_edges) (cf. report).
        num_cells (int).
        cell_size_x, cell_size_y (float): Size of the cells.
    """
    
    # Get the bounding box of the shapefile
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # Number and Size of the cells
    n_cells = 60
    cell_size_x = (xmax - xmin) / n_cells
    cell_size_y = (ymax - ymin) / n_cells
    
    # Projection of the grid
    crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

    # Create the cells in a loop and stock them in a list grid_cells
    grid_cells = []
    for i in range(n_cells):
        for j in range(n_cells):
            x0 = xmin + i * cell_size_x
            y0 = ymin + j * cell_size_y
            x1 = x0 + cell_size_x
            y1 = y0 + cell_size_y
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    
    # Create a directed graph using NetworkX
    G = nx.DiGraph()

    # Add nodes to the graph using their position
    for i, posi in pos_coords.iterrows():
        G.add_node(posi['POS'], pos=(posi['Longitude'], posi['Latitude']))
    
    # Manually add edges to the graph
    # Clockwise
    G.add_edge('1', '2', weight=0.1)
    G.add_edge('2', '3', weight=0.1)
    G.add_edge('3', '4', weight=0.1)
    G.add_edge('4', '5', weight=0.1)
    G.add_edge('5', '6', weight=0.1)

    # Anti Clockwise
    G.add_edge('7', '8', weight=0.1)
    G.add_edge('8', '9', weight=0.1)
    G.add_edge('8', '9L', weight=0.1)
    G.add_edge('9', '10', weight=0.1)
    G.add_edge('10', '11', weight=0.1)

    pos = nx.get_node_attributes(G, 'pos')

    
    # Plot (shapefile + grid cells)
    # Plot the roads map (shapefile)
    plt.figure(figsize=(10, 8))
    ax = gdf.plot(color='blue')
    
    # Plot the grid cells
    cell.plot(ax=ax, facecolor="none", edgecolor='grey')

    # Plot scanners as red crosses
    plt.scatter(scanner_coords['Longitude'], scanner_coords['Latitude'], s=80, marker='x', color='red', label='Scanners', zorder=2)
    
    # Plot graph G (different figure)
    plt.figure(figsize=(15, 12))
    # Draw nodes
    plt.scatter(pos_coords['Longitude'], pos_coords['Latitude'], s=80, marker='s', color='blue')
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2.0, arrows=True)

    plt.show()
    
    # Initialize a distance matrix with zeros of size (num_cells, num_cells, num_edges)
    num_cells = n_cells
    num_edges = len(G.edges)
    distance_matrix = np.zeros((num_cells, num_cells, num_edges))

    # Loop through each edge in the graph
    for i, (source, target) in enumerate(G.edges):
        source_x, source_y = pos_coords.loc[pos_coords['POS'] == source, ['Longitude', 'Latitude']].values[0]
        target_x, target_y = pos_coords.loc[pos_coords['POS'] == target, ['Longitude', 'Latitude']].values[0]

        # Loop through each cell in the grid
        k = 0
        l = 0
        for j, cell in enumerate(grid_cells):
            cell_geom = cell
            # Check for intersection with the cell
            if cell_geom.intersects(shapely.geometry.LineString([(source_x, source_y), (target_x, target_y)])):
                intersection = cell_geom.intersection(shapely.geometry.LineString([(source_x, source_y), (target_x, target_y)]))
                if intersection.length > 0:
                    # Compute the distance for the portion of the edge within the cell, using haversine_distance()
                    portion_length = haversine_distance(intersection.coords[0][1], intersection.coords[0][0], intersection.coords[1][1], intersection.coords[1][0])
                    distance_matrix[k, l, i] = portion_length
            k += 1
            if k >= num_cells:
                k = 0
                l += 1
    
    return G, distance_matrix, num_cells, cell_size_x, cell_size_y



def source_term(data, G, distance_matrix, gamma):
    """
    Calculate Source term (cf. report).
    
    Parameters:
        data (Pandas DataFrame).
        G (NetworkX graph): Graph of the roads network.
        distance_matrix (numpy array): Distances matrix A of size (num_cells, num_cells, num_edges) (cf. report).
        gamma (float): Emission factor (mg/m).
        
    Returns:
        source_term (list of numpy arrays): Source term for all January (length=31).
    """
    
    edge_to_scanner = {1: 'M25/5230A', 2: 'M25/5232A', 3: 'M25/5235A', 4: 'M25/5239A', 5: 'M25/5246A',
                       6: 'M25/5243B', 7: 'M25/5238B', 8: 'M25/5240L', 9: 'M25/5235B', 10: 'M25/5231B'}
    
    # Source term S(t) for each day of January where t is 15-minute intervals
    source_term = []
    
    # All January
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    current_date = start_date
    
    # Copy the dataset and add a column 'S(t)'
    data_gamma = data.copy()
    data_gamma['S(t)'] = np.zeros(data_gamma.shape[0])
    
    matrix_dict = {}
    matrix_list = []
    
    for i, _ in enumerate(G.edges):
        matrix_dict[edge_to_scanner[i+1]] = gamma *  sp.csc_matrix(distance_matrix[:,:, i])
        
    for _, row in data_gamma.iterrows():
        matrix_list.append(matrix_dict[row['Site Name']])


    data_gamma['S(t)'] = data_gamma['Rho (veh/meter)'] * matrix_list
     
    
    # Compute S(t) as explained in the report
    while current_date <= end_date:
        fixed_day_data = data_gamma[data_gamma['Report Date'] == current_date]

        # Group the data by the 'Time Period' and sum the values
        summed_data = fixed_day_data.groupby(['Time Period Ending'])['S(t)'].sum().reset_index()
        
        source_term.append(summed_data)
        
        # Move to the next day
        current_date += pd.Timedelta(days=1)

    
    return source_term


def apply_neumann_bc(c, n_cells):
    """
    Apply Neumann boundary conditions to the concentration array.

    Parameters:
        c (numpy array): Concentration array.
        n_cells (int): Number of cells in the grid.

    Returns:
        c (numpy array): Concentration array with Neumann boundary conditions applied.
    """
    
    # Left and Right boundaries
    for i in range(n_cells):
        c[i * n_cells] = c[i * n_cells + 1]
        c[(i + 1) * n_cells - 1] = c[(i + 1) * n_cells - 2]

    # Top and bottom boundaries
    for j in range(n_cells):
        c[(n_cells - 1) * n_cells + j] = c[(n_cells - 2) * n_cells + j]
        c[j] = c[j + n_cells]

    return c


def pollution_estimation(days, wind_data, source_term, n_cells, dx_meters, dy_meters, D):
    """
    Estimate the concentration of pollution at each cell in the grid.
    
    Parameters:
        days (Dictionary): Days for which we want the estimation.
        wind_data (Pandas DataFrame).
        source_term (list of numpy arrays): Source term for all January (length=31).
        n_cells (float): Number of cells in the grid.
        dx_meters (float): Size x of a cell in meters.
        dy_meters (float): Size y of a cell in meters.
        D (float): Effective diffusion coefficient (m^2/s).
        
    Returns:
        c_history (list of list of numpy arrays): Concentrations at each time and each location.
    """
    
    ndays = len(days)
    c_history = [[] for _ in range(ndays)]

    dt = 0.25 # Time step
    eye_csc = sp.eye(n_cells*n_cells, format='csc')
    
    c0 = np.zeros(n_cells*n_cells) # Suppose concentration is 0 at time 0
    
    # Loop over the days
    for i, day in days.items():
        
        specific_date = pd.to_datetime(day)
        wind_data_specific_date = wind_data[wind_data['date'].dt.date == specific_date.date()] # Keep wind data for the current day
        source_term_day = source_term[i] # Keep source term for the current day
        
        epoch_bar_t = tqdm(range(96), position=0, leave=True)
        
        # Loop over the 15 minutes intervals
        for t in epoch_bar_t:
            
            epoch_bar_t.set_description(f'Day {day} | {i+1}/{ndays}')
            
            # Determine wind and compute Diffusion and Advection matrices hourly
            if t % 4 == 0:
                
                # Determine wind in x and y directions
                    # Notice at how a wind rose is oriented (Clockwise and 0° is North) compared to a trigonometric circle
                    # So we add a pi/2 to both w1 and w2 and a minus sign to w2
                w1 = wind_data_specific_date.iloc[t//4]['windspeed'] * np.cos(np.deg2rad(wind_data_specific_date.iloc[t//4]['winddir']) + np.pi/2)
                w2 = - wind_data_specific_date.iloc[t//4]['windspeed'] * np.sin(np.deg2rad(wind_data_specific_date.iloc[t//4]['winddir']) + np.pi/2)
                
                #Diffusion term
                D_diff = D #+ D*(wind_data_specific_date.iloc[t//4]['windspeed'])**2
                alpha = D_diff * dt / dx_meters**2
                beta = D_diff * dt / dy_meters**2
                
                B_csc = sp.diags([beta/2, alpha/2, -(alpha+beta), alpha/2, beta/2],
                          [-n_cells, -1, 0, 1, n_cells],
                          shape=(n_cells*n_cells, n_cells*n_cells), format='csc')
                
                
                #Advection term
                alpha = w1 * dt / dx_meters
                beta = w2 * dt / dy_meters
                
                if w1 > 0:
                    if w2 > 0:
                        A_csc = sp.diags([beta, -(alpha+beta), alpha],
                                         [-n_cells, 0, -1],
                                         shape=(n_cells*n_cells, n_cells*n_cells), format='csc')
                    else:
                        A_csc = sp.diags([beta, (-alpha+beta), -alpha],
                                         [-n_cells, 0, 1],
                                         shape=(n_cells*n_cells, n_cells*n_cells), format='csc')
                else:
                    if w2 > 0:
                        A_csc = sp.diags([-beta, (alpha-beta), alpha],
                                         [n_cells, 0, -1],
                                         shape=(n_cells*n_cells, n_cells*n_cells), format='csc')
                    else:
                        A_csc = sp.diags([-beta, (alpha+beta), -alpha],
                                         [n_cells, 0, 1],
                                         shape=(n_cells*n_cells, n_cells*n_cells), format='csc')
                
                A_id = eye_csc + A_csc
                B_id = eye_csc + B_csc
                B_minid = eye_csc - B_csc
                
                # Check if B_id and B_minid are positive definite, in order to use sp.linalg.cg()
                eigenvalues_B_id = sp.linalg.eigsh(B_id, return_eigenvectors=False)
                eigenvalues_B_minid = sp.linalg.eigsh(B_minid, return_eigenvectors=False)
                
                if not np.all(eigenvalues_B_id > 0):
                    raise ValueError("B_id is not positive definite")
                if not np.all(eigenvalues_B_minid > 0):
                    raise ValueError("B_minid is not positive definite")

            
            # Algorithm
            k_end = round(900 / dt)
            for k in range(k_end):
                c_diffusion, info = sp.linalg.cg(B_minid, B_id @ c0)
                if info != 0:
                    raise ValueError("CG solver did not converge")
                c_advection = A_id @ c_diffusion
                c_new = c_advection + dt * source_term_day.iloc[t]['S(t)'].toarray().flatten()

                # Apply Neumann boundary conditions
                c_new = apply_neumann_bc(c_new, n_cells)
                
                c0 = c_new

            c_history[i].append(np.flip(np.array(c_new).reshape(n_cells, n_cells), axis=0))
            
    return c_history



def plot_heatmap_day(days, concentration_matrix, wind, max_windspeed, day, interval):
    """
    Plot heatmap of the concentrations for a specific day.
    
    Parameters:
        days (Dictionary).
        concentration_matrix (numpy array): Matrix of the concentrations at each location and time.
        wind (Pandas DataFrame): Wind info for specific day and time.
        day (int): Index of the day for which we want to plot the concentration.
        interval (int): Index of the time for which we want to plot the concentration.
    """
    
    plt.figure(figsize=(10, 8))
    
    # Plot the road map with a lower zorder value
    gdf.plot(ax=plt.gca(), color='blue', zorder=1)
    
    # Overlay the pollution heatmap with a higher zorder value
    extent = [gdf.total_bounds[0], gdf.total_bounds[2], gdf.total_bounds[1], gdf.total_bounds[3]]
    plt.imshow(concentration_matrix, cmap='Reds', alpha=0.7, extent=extent, zorder=2)
    plt.colorbar()
    
    # Add wind arrow on the map
        # Notice at how a wind rose is presented (Clockwise and 0° is North) compared to a trigonometric circle
        # So we add a pi/2 to both w1 and w2 and a minus sign to w2
    w1 = wind['windspeed'] * np.cos(np.deg2rad(wind['winddir']) + np.pi/2)
    w2 = - wind['windspeed'] * np.sin(np.deg2rad(wind['winddir']) + np.pi/2)
    
    scale_factor = 3.6
    arrow_scale = max_windspeed * scale_factor
    head_width_factor = 1.1 * scale_factor
    head_length_factor = 1.7 * scale_factor
    
    head_width = head_width_factor
    head_length = head_length_factor
    
    plt.quiver(gdf.total_bounds[2] - 0.08 * (gdf.total_bounds[2] - gdf.total_bounds[0]),
               gdf.total_bounds[3] - 0.2 * (gdf.total_bounds[3] - gdf.total_bounds[1]),
               w1, w2,
               scale=arrow_scale,
               color='black',
               headwidth=head_width, headlength=head_length,
               pivot='mid',
               zorder=3)
    
    
    # Set description
    minutes = (interval+1) * 15
    h = minutes / 60
    hours = int(h)
    minutes = int((h - hours) * 60)
    if minutes == 0:
        plt.title(f'NO$_2$ Concentration Heatmap on {days[day]} at {hours}h')
    else:
        plt.title(f'NO$_2$ Concentration Heatmap on {days[day]} at {hours}h{minutes}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.show()


def plot_heatmap(days, c_history, wind_data):
    """
    Plot heatmap of the concentrations for multiple consecutive day.
    
    Parameters:
        days (Dictionary).
        c_history (list of list of numpy arrays): Concentrations at each time and each location.
        wind_data (Pandas DataFrame).
    """
    
    # Loop over the days
    for i, day in days.items():
        specific_date = pd.to_datetime(day)
        wind_day = wind_data[wind_data['date'].dt.date == specific_date.date()] # Keep wind data for the current day
        # Loop over the 15 minutes intervals
        for j in range(96):
            plot_heatmap_day(days, c_history[i][j], wind_day.iloc[int(j/4)], max(wind_data['windspeed']), i, j)


def c_coords_history(days, longitude, latitude, c_history, dx, dy, df_traffic, df_wind):
    """
    Plot the concentrations over time at a fixed location.
    
    Parameters:
        days (Dictionary).
        longitude (float).
        latitude (float).
        c_history (list of list of numpy arrays): Concentrations at each time and each location.
        dx, dy (float): Size of a cell (in coordinates).
    """
    
    number_days = len(days)
    number_intervals_per_day = 96
    total_intervals = number_intervals_per_day * number_days

    start_date = pd.to_datetime(list(days.values())[0])

    # Find the coordinates of the closest cell to the given coordinates 
    cell_x = int((longitude - xmin) / dx)
    cell_y = int((latitude - ymin) / dy)
    
    # Get the concentrations for this cell over time
    pollution_values = [c_history[i][time][cell_x, cell_y] for i, _ in days.items() for time in range(96)]
    
    # Plots
        # Location on the map
    plt.figure(figsize=(10, 8))
    gdf.plot(ax=plt.gca(), color='blue', zorder=1)
    plt.scatter(longitude, latitude, s=100, marker='x', color='red', label='Position')
    plt.title(f'Position on the map at\n Lon={longitude} | Lat={latitude}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()
    
        # Concentration of pollution
    plt.figure(figsize=(10, 8))
    plt.plot(pollution_values, c='brown')
    plt.title(f'NO$_2$ Concentration at\n Lon={longitude} | Lat={latitude}')
    plt.ylabel(r'NO$_2$ Concentration ($\mu g/m^3$)')
    
    # Redefine x-axis
    interval_ticks = np.arange(1, total_intervals + 1, number_intervals_per_day)
    date_range = pd.date_range(start=start_date, periods=number_days, freq='D')
    date_labels = [date.strftime('%Y-%m-%d') for date in date_range]
    plt.xticks(interval_ticks, date_labels, rotation=45)
    
    # Add max and min pollution values
    max_value = np.max(pollution_values)
    min_value = np.min(pollution_values)
    plt.text(total_intervals, max_value, f'Max: {max_value:.3f}', ha='right', va='top', color='red')
    plt.text(total_intervals, min_value, f'Min: {min_value:.3f}', ha='right', va='bottom', color='blue')
    
    plt.grid()
    plt.show()
    
        # CO Concentration / Wind / Traffic flow for a week
    start_date = '2023-01-05'
    end_date = '2023-01-12'
    
    df_wind_week = df_wind[(df_wind['date'] >= start_date) & (df_wind['date'] <= end_date)]
    
    fig, axes = plt.subplots(nrows=3, ncols=1)
    
    # Wind Speed Plot
    axes[0].plot(df_wind_week['date'], df_wind_week['windspeed'], color='blue')
    axes[0].set_title('Wind Speed')
    axes[0].axes.get_xaxis().set_visible(False)
    axes[0].axes.get_yaxis().set_visible(False)
    
    # Pollution Concentration Plot
    start_index = datetime.strptime(start_date, '%Y-%m-%d').day - 1
    end_index = datetime.strptime(end_date, '%Y-%m-%d').day - 1
    
    pollution_values = [c_history[i][time][cell_x, cell_y] for i in (start_index, end_index+1) for time in range(96)]
    
    axes[1].plot(pollution_values, color='green')
    axes[1].set_title('NO$_2$ Concentration from Traffic Data')
    axes[1].axes.get_xaxis().set_visible(False)
    axes[1].axes.get_yaxis().set_visible(False)
    
    # Traffic Plot
    traffic_flow(df_c, 'M25-Clockwise', start_date=start_date, end_date=end_date, ax=axes[2], descriptions=False)
    axes[2].set_title('Traffic Flow')
    axes[2].axes.get_yaxis().set_visible(False)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




# ------- MAIN ------- #


# Scanners coordinates
scanner_coords = pd.DataFrame({'Scanner': ['M25/5230A', 'M25/5232A', 'M25/5235A', 'M25/5239A', 'M25/5246A',
                                           'M25/5231B', 'M25/5235B', 'M25/5238B', 'M25/5240L', 'M25/5243B'],
                               'Longitude': [-0.38214, -0.37864, -0.37398, -0.36796, -0.35760,
                                             -0.38061, -0.37409, -0.36969, -0.36686, -0.36232],
                               'Latitude': [51.71652, 51.71639, 51.71578, 51.71414, 51.71102,
                                            51.71635, 51.71564, 51.71450, 51.71347, 51.71214]})


# Nodes coordinates
pos_coords = pd.DataFrame({'POS': ['1', '2', '3', '4', '5', '6',
                                   '7', '8', '9', '9L', '10', '11'],
                           'Longitude': [-0.38214, -0.37864, -0.37398, -0.371307, -0.363183, -0.354869,
                                          -0.354966, -0.362935, -0.373797, -0.370542, -0.382288, -0.384595],
                           'Latitude': [51.71652, 51.71639, 51.71578, 51.715141, 51.712622, 51.710472,
                                          51.710305, 51.712311, 51.715582, 51.714427, 51.716397, 51.716290]})


# Wanted days
days = {0: '2023-01-01', 1: '2023-01-02', 2: '2023-01-03', 3: '2023-01-04', 4: '2023-01-05', 5: '2023-01-06',
        6: '2023-01-07', 7: '2023-01-08', 8: '2023-01-09', 9: '2023-01-10', 10: '2023-01-11', 
        11: '2023-01-12', 12: '2023-01-13', 13: '2023-01-14', 14: '2023-01-15', 15: '2023-01-16',
        16: '2023-01-17', 17: '2023-01-18', 18: '2023-01-19', 19: '2023-01-20', 20: '2023-01-21',
        21: '2023-01-22', 22: '2023-01-23', 23: '2023-01-24', 24: '2023-01-25', 25: '2023-01-26',
        26: '2023-01-27', 27: '2023-01-28', 28: '2023-01-29', 29: '2023-01-30', 30: '2023-01-31'}



# -- Clockwise -- #

print('CLOCKWISE:\n')
data_c = pd.read_csv('M25-Clockwise.csv')
df_c = clean_data(data_c)

# Availability for each scanner
availability(df_c, 'M25-Clockwise')

# Interpolate NA values
df_c = interpolate_na(df_c)

# Add needed features
add_features(df_c)

# Traffic flow for a week
traffic_flow(df_c, 'M25-Clockwise', start_date='2023-01-16', end_date='2023-01-22')

# Traffic flow on random days in January
fig, plots = plt.subplots(2, 3, figsize=(15,10), tight_layout=True)
traffic_flow(df_c, 'M25-Clockwise', 'C Monday 16/01/23', start_date='2023-01-16', end_date='2023-01-16', ax=plots[0,0], descriptions=False)
traffic_flow(df_c, 'M25-Clockwise', 'C Wednesday 18/01/23', start_date='2023-01-18', end_date='2023-01-18', ax=plots[0,1], descriptions=False)
traffic_flow(df_c, 'M25-Clockwise', 'C Thursday 12/01/23', start_date='2023-01-12', end_date='2023-01-12', ax=plots[0,2], descriptions=False)
traffic_flow(df_c, 'M25-Clockwise', 'C Friday 20/01/23', start_date='2023-01-20', end_date='2023-01-20', ax=plots[1,0], descriptions=False)
traffic_flow(df_c, 'M25-Clockwise', 'C Saturday 21/01/23', start_date='2023-01-21', end_date='2023-01-21', ax=plots[1,1], descriptions=False)
traffic_flow(df_c, 'M25-Clockwise', 'C Sunday 29/01/23', start_date='2023-01-29', end_date='2023-01-29', ax=plots[1,2], descriptions=False)
plt.suptitle('Traffic flow (Clockwise) scanned by each scanner on random days in January')
plt.tight_layout()
plt.show()


# -- Anti-clockwise -- #

print('\nANTI-CLOCKWISE:\n')
data_ac = pd.read_csv('M25-Anti-clockwise.csv')
df_ac = clean_data(data_ac)

# Availability for each scanner
availability(df_ac, 'M25-Anti-clockwise')

# Interpolate NA values
df_ac = interpolate_na(df_ac)

# Add needed features
add_features(df_ac)

# Traffic flow for a week
traffic_flow(df_ac, 'M25-Anti-clockwise', start_date='2023-01-16', end_date='2023-01-22')

# Traffic flow on random days in January
fig, plots = plt.subplots(2, 3, figsize=(15,10), tight_layout=True)
traffic_flow(df_ac, 'M25-Anti-clockwise', 'AC Monday 16/01/23', start_date='2023-01-16', end_date='2023-01-16', ax=plots[0,0], descriptions=False)
traffic_flow(df_ac, 'M25-Anti-clockwise', 'AC Wednesday 18/01/23', start_date='2023-01-18', end_date='2023-01-18', ax=plots[0,1], descriptions=False)
traffic_flow(df_ac, 'M25-Anti-clockwise', 'AC Thursday 12/01/23', start_date='2023-01-12', end_date='2023-01-12', ax=plots[0,2], descriptions=False)
traffic_flow(df_ac, 'M25-Anti-clockwise', 'AC Friday 20/01/23', start_date='2023-01-20', end_date='2023-01-20', ax=plots[1,0], descriptions=False)
traffic_flow(df_ac, 'M25-Anti-clockwise', 'AC Saturday 21/01/23', start_date='2023-01-21', end_date='2023-01-21', ax=plots[1,1], descriptions=False)
traffic_flow(df_ac, 'M25-Anti-clockwise', 'AC Sunday 29/01/23', start_date='2023-01-29', end_date='2023-01-29', ax=plots[1,2], descriptions=False)
plt.suptitle('Traffic flow (Anti-clockwise) scanned by each scanner on random days in January')
plt.tight_layout()
plt.show()



# -- Wind and Pollution data -- #

# Wind data
wind_data = pd.read_csv("Bricket Wood 2023-01-01 to 2023-01-31.csv")
wind_data.rename(columns={'datetime': 'date',
                         }, inplace=True)
wind_data['date'] = pd.to_datetime(wind_data['date'])
# Plot wind data
plot_wind(wind_data, 'Visualcrossing')

wind_data['windspeed'] = wind_data['windspeed'] / 3.6 # Windspeed to m/s


# Pollution data
df_pollution = pd.read_csv('air_quality.csv', delimiter=';')
df_pollution['End Date'] = pd.to_datetime(df_pollution['End Date'], format='%d/%m/%Y')


plots_wind_pol_traffic(df_c, wind_data, df_pollution, start_date='2023-01-16', end_date='2023-01-22')



# --- Model --- #

# Create and plot: Map + Grid + Graph
fp = "M25_section_line.shp"
gdf = gpd.read_file(fp)
xmin, ymin, xmax, ymax = gdf.total_bounds

G, distance_matrix, n_cells, dx, dy = shape_grid_graph(gdf, scanner_coords, pos_coords)

# Convert cell sizes dx and dy in meters
reference_point = (ymin, xmin)
reference_point_lat = reference_point[0]
dx_meters = geopy.distance.geodesic((reference_point_lat, xmin), (reference_point_lat, xmin + dx)).meters
dy_meters = geopy.distance.geodesic((ymin, reference_point[1]), (ymin + dy, reference_point[1])).meters


# Source term
stacked_data = pd.concat([df_c, df_ac], ignore_index=True)
gamma = 150 #0.016 # CO emission factor for a small Diesel passenger car (mg/m)
              #150 #NOx emission factor (micro_g/m)
source_term = source_term(stacked_data, G, distance_matrix, gamma)


# Pollution estimation
D = 10.0 # Parameter for the effective diffusion coefficient
c_history = pollution_estimation(days, wind_data, source_term, n_cells, dx_meters, dy_meters, D)



# -- Results Plots -- #

# Heatmap of pollution
plot_heatmap(days, c_history, wind_data)


# Pollution at a fixed location over time
longitude = -0.37
latitude = 51.715
c_coords_history(days, longitude, latitude, c_history, dx, dy, df_c, wind_data)
