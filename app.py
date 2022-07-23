import os
import glob
import math
import json
import plotly
import folium
import calendar
import warnings
import requests
import numpy as np
import pandas as pd
import numpy.ma as ma
import streamlit as st
from utide import solve
import matplotlib as mpl
import cartopy.crs as ccrs
from folium import plugins
import plotly.express as px
import plotly.subplots as sp
from genericpath import isdir
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from modules import tidal_ellipse
warnings.filterwarnings("ignore")
plotly.offline.init_notebook_mode()
import sort_dataframeby_monthorweek
import cartopy.io.img_tiles as cimgt
from modules import plot_single_rose
from matplotlib.dates import date2num
from modules import put_tides_on_folium
from numpy import arange, empty, matlib
from streamlit_folium import folium_static
from IPython.core.display import display, HTML
from modules import depth_avg_prof_stats as dps
from matplotlib import colors as mcolors, legend, style 
# ==============================================================
display(HTML("<style>.container { width:110% !important; }</style>"))
from netCDF4 import Dataset # http://unidata.github.io/netcdf4-python/
st.set_page_config(page_title="Tidal Stream", page_icon="🌊", layout="wide")
# ------------------------------------------------------------------------------


# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# # Sort colors by hue, saturation, value and name.
# by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
#                 for name, color in colors.items())
# sorted_names = [name for hsv, name in by_hsv]


st.markdown(f'<h1 style="color: orange;border-radius:50%;" >Tidal Stream</h1>', unsafe_allow_html=True)

st.markdown(
    f'<a style="color: pink;border-radius:50%;" >An Interactive Tool for ADCP Data Verification: Tidal Analysis, '
    f'PCA and more ...</a>',
    unsafe_allow_html=True)


def version_info():
    # Rule: Version: <Major>.<Minor>.<Patch/Upgrade>
    version = '0.0.0'
    st.markdown(
        f'''<a style="color:pink;font-size:12px;border-radius:0%;">Version: v{version}</a>''',
        unsafe_allow_html=True)


version_info()

data_directories = glob.glob(os.getcwd() + '/Data/*')
dir_dict = {f"{d.split('/')[-1]}": d + f"/{d.split('/')[-1]}-*/*.nc" for d in data_directories if isdir(d)}
loc_and_dep_cont = st.container()
with loc_and_dep_cont:
    directory_to_work_with = st.selectbox("Select Location", ["Select Location"] + list(dir_dict.keys()), index=0)

year_month_selectors_container = st.container()

# ------------------------------------------------------------------------------
if directory_to_work_with in dir_dict.keys():  # directory_to_work_with is the same as location code

    loc_based_file_paths = glob.glob(dir_dict.get(directory_to_work_with))
    loc_based_file_paths.sort()
    st.header(directory_to_work_with)

    # ------------------------------------------------------------------------------

    fnms = []
    for fnm in loc_based_file_paths:
        fnms.append(os.path.basename(fnm))
    # ------------------------------------------------------------------------------

    list_of_files_used = pd.DataFrame({'Hourly data files': fnms})


    def get_site_info(loc_based_file_paths_):
        details_ = pd.DataFrame()

        for filename in loc_based_file_paths_:
            # st.write(filename)
            fid = Dataset(filename, mode='r')
            nctime = fid.variables['time'][:]  # get values
            if len(nctime) > 3 * 30 * 24:  # include only files with 90 days worth of hourly averaged data (3*30*24)
                try:
                    if 'Nortek' in fid.device_name.split():  # Nortek
                        bd = fid.adcp_setup_blanking_distance_meters
                    else:  # RDI
                        bd = fid.adcp_setup_WE_blanking_distance_meters
                except AttributeError:
                    # we could have adcps with no blanking distance set(?)
                    bd = 0

                try:
                    if 'Nortek' in fid.device_name.split():  # Nortek
                        cs = fid.adcp_setup_cell_size_meters
                    else:  # RDI
                        cs = fid.adcp_setup_WS_cell_size_meters
                except AttributeError:
                    # we can't have adcps with no cell size set but to avoid some weird errors
                    cs = 0

                try:
                    if 'Nortek' in fid.device_name.split():  # Nortek
                        nc = fid.adcp_setup_number_cells
                    else:  # RDI
                        nc = fid.adcp_setup_WN_number_bins
                except AttributeError:
                    # if bumber of cells is not found, we assume 1
                    nc = 1

                details1 = pd.DataFrame({"site_name": [fid.site_name],
                                         "begin": [fid.time_coverage_start],
                                         "end": [fid.time_coverage_end],
                                         "device_id": [fid.device_id],
                                         "device_heading": [fid.device_heading],
                                         "orientation": [fid.orientation],
                                         "platform_depth": [fid.platform_depth],
                                         "device_name": [fid.device_name],
                                         "location_name": [fid.location_name],
                                         "latitude": [fid.variables['latitude'][:].data[0]],
                                         "longitude": [fid.variables['longitude'][:].data[0]],
                                         "blanking_distance(m)": [bd],
                                         "cell_size(m)": [cs],
                                         "number_of_depth_cells": [nc],
                                         })

                details_ = pd.concat([details_, details1], ignore_index=True)
        df = details_.copy()
        df[['begin', 'end']] = df[['begin', 'end']].applymap(lambda x: pd.Timestamp(x) if x is not None else None)

        df.loc[df.end.values == None, 'end'] = pd.Timestamp(datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'))  # this
        # comparison is not pythonic but works
        details_['number_of_days'] = (df.end - df.begin).dt.days
        details_ = details_[
            ['site_name', 'begin', 'end', 'number_of_days', 'device_heading', 'blanking_distance(m)', 'cell_size(m)',
             'number_of_depth_cells', 'orientation', 'platform_depth', 'latitude', 'longitude', 'device_id',
             'device_name', 'location_name']]

        details_[
            ['device_heading', 'blanking_distance(m)', 'cell_size(m)', 'platform_depth', 'latitude', 'longitude']] = \
            details_[['device_heading', 'blanking_distance(m)', 'cell_size(m)', 'platform_depth', 'latitude',
                      'longitude']].astype(float).round(2)
        details_[['number_of_days', 'number_of_depth_cells', 'device_id']] = details_[
            ['number_of_days', 'number_of_depth_cells', 'device_id']].astype(int)
        details_.sort_values(by=['site_name'], inplace=True, ascending=False)
        return details_


    details = get_site_info(loc_based_file_paths)


    def get_data(loc_based_file_paths_):
        site_names_ = []
        sites_with_at_least_1month_data_ = []
        all_angles_ = ['']
        all_us_ = ['']
        all_vs_ = ['']
        all_ws_ = ['']
        all_t_ = ['']
        all_z_ = ['']

        for filename in loc_based_file_paths_:
            fid = Dataset(filename, mode='r')
            site_names_.append(fid.site_name)
            nctime = fid.variables['time'][:]  # get values
            if len(nctime) > 3 * 30 * 24:  # include only files with 90 days worth of hrly data (3*30*24)
                sites_with_at_least_1month_data_.append(fid.site_name)

                #         print("file used: ",filename)
                # t_unit = fid.variables['time'].units # get unit  'days since 19700101T000000Z'
                t = pd.to_datetime(nctime.data, unit='d')
                # t_cal = fid.variables['time'].calendar
                var1name = 'u'
                u = fid.variables[var1name][:]
                var2name = 'v'
                v = fid.variables[var2name][:]
                var3name = 'depth'
                Z = fid.variables[var3name][:]
                var4name = 'w'
                w = fid.variables[var4name][:]
                # take care of missing data
                u.data[u.data == -9999999.0] = np.nan
                v.data[v.data == -9999999.0] = np.nan
                w.data[w.data == -9999999.0] = np.nan
                Z[Z == -9999999.0] = np.nan

                us = [u.data[:, k] for k in range(len(Z))]
                vs = [v.data[:, k] for k in range(len(Z))]
                ws = [w.data[:, k] for k in range(len(Z))]

                angles = [list(map(math.atan2, u.data[:, k], v.data[:, k])) for k in range(len(Z))]
                # speed
                # ws = [np.sqrt(u.data[:,k]**2 + v.data[:,k]**2 ) for k in range(len(Z))]
                all_angles_.append(angles)
                all_us_.append(us)
                all_vs_.append(vs)
                all_ws_.append(ws)
                all_t_.append(t)
                all_z_.append(Z)

        # remove first (empty element- this is due to the initialization with  empty element... )

        all_angles_.pop(0)
        all_us_.pop(0)
        all_vs_.pop(0)
        all_ws_.pop(0)
        all_t_.pop(0)
        all_z_.pop(0)

        return site_names_, sites_with_at_least_1month_data_, all_angles_, all_us_, all_vs_, all_ws_, all_t_, all_z_


    site_names, sites_with_at_least_1month_data, all_angles, all_us, all_vs, all_ws, all_t, all_z = get_data(
        loc_based_file_paths)


    zi = st.selectbox("Select depth bin* to plot time series data", list(range(
        len(all_z[0]))))  # this won't work all the time since some deployment sites have different number of depth bins

    st.markdown(
    f'''<a style="color:orange;font-size:16px;border-radius:0%;">*note that bin numbers may not necessarily represent the same actual depth across all deployment site. 
                This can be checked in the metadata for each deployment site on this page.</a>''',
    unsafe_allow_html=True)

    testdata = pd.DataFrame({'u': all_us[0][zi], 'v': all_vs[0][zi], 'w': all_ws[0][zi]}, index=all_t[0])
    for k in range(1, len(all_t)):
        testdata = pd.concat(
            [testdata, pd.DataFrame({'u': all_us[k][zi], 'v': all_vs[k][zi], 'w': all_ws[k][zi]}, index=all_t[k])],
            axis=0)
        testdata.index.name = 'time'

    test_plot = px.line(testdata, x=testdata.index, y=["u", "v", "w"],
                        title=f"Data from all deployments, depth bin {zi}",
                        color_discrete_sequence=px.colors.qualitative.Dark24)
    test_plot.update_layout(xaxis_title="Time", yaxis_title="Velocity (m/s)", hovermode="x")
    st.plotly_chart(test_plot, use_container_width=True)



# @st.cache
def depth_average(all_z_, all_us_, all_vs_, all_ws_, all_t_, all_angles_, deployment_idx_, n1, n2):
    depth_avg_us_ = np.nanmean(all_us_[deployment_idx_][n1:n2 + 1][:], axis=0)
    depth_avg_vs_ = np.nanmean(all_vs_[deployment_idx_][n1:n2 + 1][:], axis=0)
    # I know avg of ws and angles should not be done this way, but good enough for now
    # since we are comparing similarly calculated values among deployments.
    #  The correct way to do this is to calculate them from depth averaged u and v.
    depth_avg_ws_ = np.nanmean(all_ws_[deployment_idx_][n1:n2 + 1][:], axis=0)
    depth_avg_angles_ = np.nanmean(all_angles_[deployment_idx_][n1:n2 + 1][:], axis=0)
    depth_avg_t_ = all_t_[deployment_idx_][:][:]  # time is the same for all depths
    depth_avg_z_ = all_z_[deployment_idx_][n1:n2 + 1]
    return depth_avg_us_, depth_avg_vs_, depth_avg_ws_, depth_avg_angles_, depth_avg_t_, depth_avg_z_


# -----------------------------------------------------------------------------------------------------------------------
# @st.cache
def calc_coef(df, site_details, deployment_idx_):
    kw = dict(
        lat=site_details.loc[deployment_idx_].latitude,
        nodal=False,
        trend=False,
        method='ols',
        conf_int='linear',
        Rayleigh_min=0.95)

    time = date2num(df.index.to_pydatetime())
    coef = solve(time, df['u'].values, df['v'].values, **kw)
    return coef


# -----------------------------------------------------------------------------------------------------------------------

# @st.cache
def get_params(df, site_detail_, deployment_idx_):
    coef = calc_coef(df, site_detail_, deployment_idx_)
    keys = ('Lsmaj', 'Lsmin', 'theta', 'g', 'g_ci', 'Lsmaj_ci', 'Lsmin_ci', 'theta_ci', 'PE', 'SNR')
    df = pd.DataFrame({k: coef.get(k) for k in keys})
    df.sort_values(by=['Lsmaj', 'Lsmin'], ascending=False, inplace=True)
    df['frq'] = coef['aux']['frq']
    df.index = coef.get('name')
    df.index.name = 'name'
    return df


@st.cache
def tidal_ellipse_points(df, constituent_name_, site_detail_, deployment_idx_, scale_=1):
    center_lat_, center_lon_ = site_detail_.loc[deployment_idx_].latitude, site_detail_.loc[deployment_idx_].longitude
    # st.write('center_lat, center_lon',type(center_lat), type(center_lon))
    location_name_ = site_detail_.loc[deployment_idx_].location_name
    a_x, b_x, theta = df.loc[constituent_name_, ['Lsmaj', 'Lsmin', 'theta']].values  # only plot first constituent (M2)

    theta_x = np.arange(-np.pi, np.pi, 0.01)
    ang_x = np.radians(-1.0 * theta)  # note sign to make it confirm to prjections on map
    rot_matrix_ = np.array([[np.cos(ang_x), np.sin(ang_x)],
                            [-np.sin(ang_x), np.cos(ang_x)]])

    ellipse_xx = np.array(a_x * np.cos(theta_x))
    ellipse_yx = np.array(b_x * np.sin(theta_x))
    ellipse_rx = scale_ * np.dot(rot_matrix_, [ellipse_xx, ellipse_yx])
    tidal_point = pd.DataFrame(
        {'longitude': ellipse_rx[0, :] + center_lon_, 'latitude': ellipse_rx[1, :] + center_lat_})
    return tidal_point, center_lat_, center_lon_, location_name_


# tidal_points,  center_lat, center_lon, location_name = tidal_ellipse_points(df_tides,details)
# -----------------------------------------------------------------------------------------------------------------------

# @st.cache
def plot_tidal_ellipse(df_tides_, constituent_name_, site_detail_, deployment_idx_, ax_, color='red', scale_=1,
                       line_width=1.5):
    df, center_lat_, center_lon_, location_name_ = tidal_ellipse_points(df_tides_, constituent_name_, site_detail_,
                                                                        deployment_idx_, scale_)
    # ax.scatter(df['longitude'],df['latitude'],s=2, label=constituent_name)

    ax_.scatter(center_lon_, center_lat_, s=12, color=color, marker='o')
    ax_.plot(df['longitude'], df['latitude'], label=constituent_name_, linewidth=line_width, color=color)
    x = df[['longitude', 'latitude']].agg(['min', 'max'])
    center = pd.DataFrame({'longitude': [center_lon_], 'latitude': [center_lat_]}, index=['center'])
    # trying to plot ellipse axes
    x = pd.concat([x, center])
    y = x.loc[['center', 'max']]

    ax_.set_xlabel('longitude')
    ax_.set_ylabel('latitude')
    ax_.set_xlim(x['longitude']['min'] - .150, x['longitude']['max'] + .150)
    ax_.set_ylim(x['latitude']['min'] - .150, x['latitude']['max'] + .150)
    ax_.set_title(f'Tidal ellipse: {location_name_}')
    return ax_, center_lat_, center_lon_, location_name_


# -----------------------------------------------------------------------------------------------------------------------
# @st.cache
def time_average(all_t_, all_z_, all_us_, all_vs_, all_ws_, all_angles_, deployment_idx_):
    rad_to_deg = np.matlib.repmat(180 / np.pi, np.shape(all_us_[deployment_idx_][:][:])[0],
                                  np.shape(all_us_[deployment_idx_][:][:])[1])
    # np.matlib will be deprecated in future numpy release
    u_t = np.nanmean(all_us_[deployment_idx_][:][:], axis=1)
    v_t = np.nanmean(all_vs_[deployment_idx_][:][:], axis=1)
    w_t = np.nanmean(all_ws_[deployment_idx_][:][:], axis=1)
    angle_t = np.nanmean(rad_to_deg * all_angles_[deployment_idx_][:][:], axis=1)
    z_t = all_z_[deployment_idx_][:][:]
    t = all_t_[deployment_idx_][:][:]
    df = pd.DataFrame({'u': u_t, 'v': v_t, 'w': w_t, 'angle': angle_t}, index=z_t,
                      columns=['u', 'v', 'w', 'angle'], dtype=float)
    # remove values for depth below surface
    df = df.loc[df.index > 0]

    return df


# @st.cache
def plot_cw_ccw(df, name, proj_=None):
    df = df.loc[name]
    sema, semi = df['Lsmaj'], df['Lsmin']
    pha, inc = df['g'], df['theta']
    ecc = semi / sema
    fig_, ax_ = tidal_ellipse.plot_ell(sema, ecc, inc, pha, label=name, proj=proj_)
    ax_.title.set_text(f'{name} {ax_.title.get_text()}')
    ax_.set_xlim(-.05, .05)
    ax_.set_ylim(-.05, .05)
    return fig_, ax_


# @st.cache
def query_deps(location_code, token=None, url='https://data.oceannetworks.ca/api/deployments'):
    assert token is not None, 'Please provide a token'
    df = pd.DataFrame({})
    for deviceCategoryCode in ['ADCP55KHZ', 'ADCP75KHZ', 'ADCP150KHZ', 'ADCP300KHZ', 'ADCP400KHZ', 'ADCP600KHZ',
                               'ADCP1200KHZ', 'ADCP1MHZ', 'ADCP2MHZ']:  # this will need updating if new ADCPs with different category codes are added

        filters = {'method': 'get',
                   'token': token,
                   # your personal token obtained from the 'Web Services API' tab at
                   # https://data.oceannetworks.ca/Profile when logged in.
                   'locationCode': location_code,
                   'deviceCategoryCode': deviceCategoryCode
                   }

        response = requests.get(url, params=filters)

        if response.ok:
            deployments = json.loads(str(response.content, 'utf-8'))  # convert the json response to an object
            for deployment in deployments:
                df = df.append(pd.DataFrame([deployment]))
        else:
            if response.status_code == 400:
                error = json.loads(str(response.content, 'utf-8'))
                st.write(error)  # json response contains a list of errors, with an errorMessage and parameter
            else:
                st.write('Error {} - {}'.format(response.status_code, response.reason))
    dff = df.copy()

    dff[['begin', 'end']] = dff[['begin', 'end']].applymap(lambda x: pd.Timestamp(x) if x is not None else None)
    return dff[['begin', 'end', 'heading', 'pitch', 'roll', 'depth', 'lat', 'lon', 'deviceCategoryCode', 'deviceCode',
                'hasDeviceData']].sort_values(by='begin', ascending=False).reset_index().drop(columns='index')


# @st.cache
def add_bathymetry(ax_, site_detail_, deployment_idx_, min_cell_number):
    bathy = '/home/zelalem/my_repos/mygithubrepos/adcp-heading-verification/miscellaneous/bathymetry' \
            '/GEBCO_01_Feb_2022_36c3a4721f1e/gebco_2021_n56.0_s45.0_w-136.0_e-122.0.nc '
    fid = Dataset(bathy, mode='r')
    lat = fid.variables['lat'][:]
    lon = fid.variables['lon'][:]
    e = fid.variables['elevation'][:]
    e_ocean = ma.masked_greater(e, 0)
    depth = site_detail_.loc[deployment_idx_]['platform_depth']
    cs = site_detail_.loc[deployment_idx_]['cell_size(m)']
    bd = site_detail_.loc[deployment_idx_]['blanking_distance(m)']
    levels = list(np.arange(-depth, (bd + min_cell_number * cs) - depth,
                            .5))  # show depth contours of only blanking distance + first 2 cell depths at .5m spacing
    st.sidebar.markdown(
        f'{len(levels)} depth contours with range {-1 * max(levels)}m -to- {-1 * min(levels)}m defined between '
        f'platform depth ({depth}) and blanking distance + first {min_cell_number} cell depth(s), at .5m spacing:')
    h = ax_.contourf(lon, lat, e_ocean, levels=levels, cmap='RdBu',
                     linewidths=0.5)  # ,colors=['r','b','g','y','m'],linewidths=0.25)
    return fig, ax_, h, e_ocean, lat, lon


def plot_bar_chart(counts_):
    """
    plots data coverage bar chart as tot hrs by months
    """
    bar = px.bar(counts_, x='Month', y='Tot. Hours')
    bar.update_layout(title_text='<b>Tot. Hours (with non-nan data) by month</b>',
                      xaxis_title='Month',
                      yaxis_title='Tot. Hours',
                      showlegend=False,
                      margin=dict(
                          l=10,
                          r=10,
                          b=50,
                          t=80,
                          pad=0
                      ),
                      width=375,
                      height=375
                      )
    return bar


def depth_contour_selector(index, column_name):
    select_min_depth_contour_ = 5
    return select_min_depth_contour_


# show_static_map = st.checkbox('Show static map using GEBCO high res bathymetry (It is slow!). If you want this
# feature, you can try to optimize the processe by selecting depth contours on the sidebar.')
zeroth_row_items_container = st.container()
first_row_plots_container = st.container()
second_row_plots_container = st.container()
third_row_plots_container = st.container()

if directory_to_work_with != "Select Location":
    folium_map = folium.Map(location=[details.loc[0]['latitude'], details.loc[0]['longitude']], zoom_start=10,
                            control_scale=True)
    with loc_and_dep_cont:

        select_deps = st.multiselect('Select Deployment(s)',
                                     options=['All'] + [details.loc[i]['site_name'] for i in range(len(details))],
                                     default=['All'],
                                     key=['All'] + [details.loc[x]['site_name'] for x in range(len(details))])
    if select_deps == ['All']:
        idxs = [i for i in range(len(details))]
    else:
        idxs = [i for i in range(len(details)) if details.loc[i]['site_name'] in select_deps]

    for deployment_idx in idxs:

        with zeroth_row_items_container:
            st.markdown(
                f'<h3 style="color: orange ;border-radius:50%;" >Metadata {(details.loc[deployment_idx]["site_name"])}</h3>',
                unsafe_allow_html=True)
            site_detail_xpdr = st.expander('Site detial', False)
            with site_detail_xpdr:
                st.write('''Metadata from netCDF files on disk that have at least 30 days of data are shown below. 
                            The number_of_days may not necessarily match actual record length. 
                            In case of mismatch between above two tables, downloading data and required. 
                            I will add a downloading script later to this app.''')
                site_detail = details.query(f"site_name == '{details.loc[deployment_idx]['site_name']}'")
                st.write(site_detail)
            with zeroth_row_items_container:
                dcol = st.columns(5)

                select_min_depth_contour = depth_contour_selector(deployment_idx, dcol[-1])
                df_t = time_average(all_t, all_z, all_us, all_vs, all_ws, all_angles, deployment_idx)
                df_t[
                    'depth'] = df_t.index.values  # add depth column to df_t temporarily for filtering data only from
                # +ve depths
                df_t = df_t.query(f"depth >= {0}").drop(
                    columns='depth')  # drop data rows with depths greater than platform depth
                dcol[0].markdown(
                    f'<a style="color: orange;border-radius:50%;" >Profile: Select min profile depth for averaging</a>',
                    unsafe_allow_html=True)
                all_depths = list(np.round(df_t.index.values, decimals=3))
                dpth1 = dcol[0].selectbox('', ['Select minimum depth '] + sorted(all_depths), index=0,
                                          key=f'{deployment_idx}')
                dcol[1].markdown(
                    f'<a style="color: orange;border-radius:50%;" >Profile: Select max profile depth for averaging</a>',
                    unsafe_allow_html=True)
                dpth2 = dcol[1].selectbox('', ['Select maximum depth'] + all_depths, index=0, key=f'{deployment_idx}')

        if dpth1 != 'Select minimum depth ' and dpth2 != 'Select maximum depth':
            if details.loc[deployment_idx].orientation == 'Up':
                n2 = pd.DataFrame({'z': all_depths}).z.tolist().index(dpth1)
                n1 = pd.DataFrame({'z': all_depths}).z.tolist().index(dpth2)
                dcol[0].markdown(f'''<a style="color: white;font-size:14px;border-radius:50%;">
                                {"Depth bins averaged: {} to {}".format(n1, n2)}</a>''',
                                 unsafe_allow_html=True)
            elif details.loc[deployment_idx].orientation == 'Down':
                n1 = pd.DataFrame({'z': all_depths}).z.tolist().index(dpth1)
                n2 = pd.DataFrame({'z': all_depths}).z.tolist().index(dpth2)
                dcol[0].write('Depth bins averaged: {} to {}'.format(n1, n2))
            else:
                dcol[0].write('no orientation info. check the data')

            depth_avg_us, depth_avg_vs, depth_avg_ws, depth_avg_angles, depth_avg_t, depth_avg_z = depth_average(all_z,
                                                                                                                 all_us,
                                                                                                                 all_vs,
                                                                                                                 all_ws,
                                                                                                                 all_t,
                                                                                                                 all_angles,
                                                                                                                 deployment_idx,
                                                                                                                 n1, n2)

            data = pd.DataFrame({"u": depth_avg_us, "v": depth_avg_vs, "w": depth_avg_ws, "angles": depth_avg_angles},
                                index=depth_avg_t,
                                columns=['u', 'v', 'w', 'angles'], dtype=float)
            data.index.name = 'time'


            def subset_data(data_, year_select, month_select):
                year_select = sorted(year_select)
                if year_select != ['All']:
                    selected_year_df = pd.DataFrame()
                    for y in year_select:
                        selected_year_df = selected_year_df.append(data_.loc[data_.index.year == int(y)])
                    data_ = selected_year_df

                if month_select == ['All']:
                    month_select = calendar.month_abbr[1:]
                    return data_
                else:
                    # subset the dataframe to the months user selected                    
                    selected_month_df = pd.DataFrame()
                    for m in month_select:
                        selected_month_df = selected_month_df.append(data_.loc[data_.index.strftime('%b') == m])
                    return selected_month_df


            # with year_month_selectors_container:
            key_digit = str(deployment_idx)
            dcol[2].markdown(f'<a style="color: orange;border-radius:50%;" >Select Years</a>', unsafe_allow_html=True)
            selected_years = dcol[2].multiselect('Select Years', ['All'] + list(set(data.index.year)), default=['All'],
                                                 key=['All'] + list(set(data.index.year)) + [key_digit])
            dcol[3].markdown(f'<a style="color: orange;border-radius:50%;" >Select Months</a>', unsafe_allow_html=True)
            selected_months = dcol[3].multiselect('Select Months', ['All'] + calendar.month_abbr[1:], default=['All'],
                                                  key=[details.loc[deployment_idx]["site_name"] + '_' + i for i in
                                                       ['All'] + calendar.month_abbr[1:]] + [key_digit])

            data = subset_data(data, selected_years, selected_months)
            if len(data) < 720 * 2:  # if the data is less than 60 days worth, notify user and wait for more data
                st.warning(
                    'Too little or no data for the selected months. Try adding more month(s). See the metadata for '
                    'more info on data availability.')
                st.stop()
            counts = data.groupby(data.index.strftime('%b')).count()  # counts non-nan values per month
            counts['Month'] = counts.index  # add month column
            counts['Tot. Hours'] = counts['u']  # add hours column
            counts = sort_dataframeby_monthorweek.Sort_Dataframeby_Month(df=counts, monthcolumnname='Month')

            fgg_bar = plot_bar_chart(counts)
            # fgg_bar.update_traces(width=1)

            data_t = time_average(all_t, all_z, all_us, all_vs, all_ws, all_angles, deployment_idx)
            df_tides = get_params(data, details, deployment_idx)  # get tidal parameters
            # -----------------------------------------------------------------------------------------------------------------------

            mpl.rcParams['axes.linewidth'] = 1
            font = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 8}
            # ----------update text-----------------
            plt.rcParams.update({
                # 'figure.facecolor': 'black',
                'font.size': 12,
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"]})

            # --------------------------------------------------
            color_list = ['black', 'blue', 'red', 'green', 'purple', 'brown', 'magenta']
            colors_dict = {f'{i}': j for i, j in zip(df_tides.index[:7], color_list)}
            # map_extent_factor = st.slider('adjust map scale', min_value=0.5, max_value=5.0, value=1.0, step=0.5)
            map_extent_factor = 1.0
            scale = .750 * map_extent_factor

            # On static map

            # Create a Stamen Terrain instance.
            stamen_terrain = cimgt.Stamen('watercolor')  # terrain-background, terrain, toner and watercolor

            # set up the plot
            proj = ccrs.PlateCarree()
            # fig, ax = plt.subplots(figsize=(8,8),dpi = 80)
            fig, ax = plt.subplots(figsize=(8, 8), dpi=60, subplot_kw=dict(projection=proj))

            # colors_dict ={'M2':'red','S2':'blue','N2':'green','K1':'maroon','O1':'purple','Q1':'brown','P1':'pink',
            # 'K2':'indigo','M4':'yellow','S4':'black'}
            for constituent_name in df_tides.index[[
                0]]:  # the constituents' magnitude are not consistent among locations. We plot only the first 3.
                # Later I will give the user the option to choose the constituents.
                ax, center_lat, center_lon, location_name = plot_tidal_ellipse(df_tides, constituent_name, details,
                                                                               deployment_idx, ax,
                                                                               color=colors_dict[constituent_name],
                                                                               scale_=scale, line_width=2)

            fig_uv_profiles = px.line(data_t, y=data_t.index, x=['u', 'v', 'w'])
            fig_uv_profiles.update_traces(line=dict(width=3))
            fig_uv_profiles.update_xaxes(title_text='current vel. (m/s)')
            fig_uv_profiles.update_yaxes(title_text='depth (m)')
            fig_uv_profiles.update_yaxes(autorange='reversed')
            fig_uv_profiles.update_layout(
                legend=dict(title_text='<b>time avg. current vel.</b>', orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1.02),
                width=350,
                height=375,
                font=dict(
                    size=12,
                ),
                margin=dict(
                    l=60,
                    r=10,
                    b=50,
                    t=60,
                    pad=0
                )
            )

            fig_uv_TS = px.line(data, x=data.index, y=['v', 'u'])

            fig_uv_TS.update_yaxes(title_text='current vel. (m/s)')
            fig_uv_TS.update_xaxes(title_text='Time')
            fig_uv_TS.update_layout(
                # legend=dict(title_text='<b>depth avg. current vel. (m/s)</b>',orientation="h"),
                legend=dict(title_text='<b>depth avg. current vel.</b>', orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0),
                font=dict(
                    size=12,
                ),
                # width=500,
                height=375,
                margin=dict(
                    l=0,
                    r=50,
                    b=0,
                    t=60,
                    pad=0
                )
            )

            # Add ellipse on folium map
            put_tides_on_folium.add_tidal_entities_on_map(df_tides, constituent_name, folium_map, center_lat,
                                                          center_lon,
                                                          f'{details.loc[deployment_idx]["site_name"]}')  # scale
            # factor of the tidal ellipse

            fgg = plot_single_rose.plot_single_currentrose(depth_avg_t, depth_avg_us, depth_avg_vs,
                                                           directory_to_work_with)
            with second_row_plots_container:
                st.markdown(
                    f'<h5 style="color: orange ;background-color:#3D0669;" >{(details.loc[deployment_idx]["site_name"])}</h5>',
                    unsafe_allow_html=True)

                TIME_SERIES_ROW = st.columns([7, 2.25, 2, 2])

                TIME_SERIES_ROW[0].plotly_chart(fig_uv_TS, use_container_width=True)
                TIME_SERIES_ROW[1].plotly_chart(fgg)
                TIME_SERIES_ROW[2].plotly_chart(fig_uv_profiles)
                TIME_SERIES_ROW[3].plotly_chart(fgg_bar)

            with third_row_plots_container:
                TIDAL_PARAMS_ROW = st.columns(1)
                TIDAL_PARAMS_ROW[0].markdown(
                    f'<a style="color: orange ;border-radius:50%;" >{(details.loc[deployment_idx]["site_name"])}: the '
                    f'top three tidal constituents for depth averaged data...</a>',
                    unsafe_allow_html=True)
                TIDAL_PARAMS_ROW[0].dataframe(df_tides.head(3))
                STATS_ROW = st.columns([2, 2])
                dap = dps.DepthAveragedProfilesStats(data, 0, -1)
                TIDAL_PARAMS_ROW[0].markdown(
                    f'<a style="color: orange ;border-radius:50%;" >{(details.loc[deployment_idx]["site_name"])}: '
                    f'Statisitcs for depth averaged data...</a>',
                    unsafe_allow_html=True)
                TIDAL_PARAMS_ROW[0].write(dap.all_stats())

    with first_row_plots_container:
        st.markdown(f'<h3 style="color: orange ;background-color:#3D0669;" >Tidal Ellipses and Currents</h3>',
                    unsafe_allow_html=True)

        colsX = st.columns([12, 2])
        with colsX[0]:
            folium.TileLayer('stamenterrain').add_to(folium_map)
            folium.TileLayer('openstreetmap').add_to(folium_map)
            folium.TileLayer('cartodbdark_matter').add_to(folium_map)
            folium.TileLayer('stamentoner').add_to(folium_map)
            folium.TileLayer('stamenwatercolor').add_to(folium_map)
            folium.TileLayer('cartodbdark_matter').add_to(folium_map)
            folium.TileLayer('stamentoner').add_to(folium_map)
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
                attr="Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC",
                name='Nat Geo Map').add_to(folium_map)
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
                name='World Imagery').add_to(folium_map)

            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}',
                attr='Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri',
                name='Ocean Basemap').add_to(folium_map)

            folium.LayerControl().add_to(folium_map)
            folium.LatLngPopup().add_to(folium_map)
            mini_map = plugins.MiniMap(toggle_display=True)
            folium_map.add_child(mini_map)
            map_height_factor = st.slider('Adjust map height by dragging the slider below', min_value=0.5,
                                          max_value=2.0, value=1.0, step=0.5)
            folium_static(folium_map, width=2700, height=400 * map_height_factor)

    st.markdown(
        f'<a style="color: orange ;border-radius:50%;" >Enter your api token to see full deployment history for {site_detail.location_name.values[0]} (requires api token)</a>',
        unsafe_allow_html=True)
    user_token = st.text_input('Token:', '', type='password')
    if user_token != '':
        st.markdown(
            f'<a style="color: orange ;border-radius:50%;" >Full deployments list for {site_detail.location_name.values[0]} (location code: {directory_to_work_with})</a>',
            unsafe_allow_html=True)
        query_deps_xpdr = st.expander('Expand to read...', False)

        with query_deps_xpdr:
            df_deps = query_deps(directory_to_work_with, user_token)

            st.write('''Note that number_of_days is the number of days between the start and end of the deployment. 
            It does not necessarily mean data are available for all these days. Sometimes devices stay in the water 
            without collecting data. Data from autonomously deployed devices may not be available at this time.''')

            st.dataframe(df_deps)

    # TODO LIST

    to_do_list_xpdr = st.expander('To do list', False)
    with to_do_list_xpdr:
        st.markdown(f'''
        <h2>To do list</h2>
        
        - [x]  Allow multiple deployments to be selected for comparison.
        
        - [x]  Consider breaking down data into seasons/months for proper comparison of deployments. For example Jul-Sept 
        flow could be quite different from the winter flow. Allow users to select seasons/months etc.

        - [ ] Add information on when ( at deployment or recovery) fixed orientation information was obtained

        - [ ] Use  PCA ellipses as the main tools going forward for decision making on the need to correct orientations.  
 
        - [ ] Keep in mind the  deep water renewal events in the SoG central that are in phase with the neap tides when 
        calculating any statistic for directions or speed. 
        
        - [ ] Collect deployment and principal flow related data for all adcp sites from Richard D. 
        
        - [ ] Add the best bathymetry (multi-beam) contour map with the exact/best estimate of where each deployment was
        and what the local “heading” is of the isobaths. While this is only a reassuring piece of information, 
        we cannot have the currents near the bottom heading into or out of the slope, they would align themselves to 
        be very close to parallel with the isobaths. So another column in the table would be the isobath heading at 
        the deployment site. 
        
        - [ ] Add PCA ellipses on a map with centres at the site lon,lat.
        
        - [ ] Build a database for long term stats so we can easily compare deployments.
        
        - [ ] Think about faster ways to calculate everything, cache everything when possible and appropriate.
        
        - [ ] Allow users to drag and align ellipses on the map, and the save new parameters. That would be a nice feature.
        
        - [ ] Add scripts to download data using the api. The scripts should be able to download data from the server and save it to 
        the local disk in the appropriate directory. If data file is to be manually updated, the directory structure should look like the one below 

        ''', unsafe_allow_html=True)

        st.image('./png/data-directory-structure-example.png', width=2000)
