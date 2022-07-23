import random
import folium
from matplotlib import colors
# from matplotlib.pyplot import contourf
import pandas as pd
import numpy as np
# from sklearn.preprocessing import scale
from math import atan2, pi


class AddOnMap(folium.FeatureGroup):
    def __init__(self, df, color, dash_array, name, center_lat, center_lon, scale=.22,**kwargs):
        self.center_lat = center_lat
        self.center_lon = center_lon  # location coordinates from metadata
        self.scale =scale
        self.point = pd.DataFrame({'longitude': self.scale *df.real.values + self.center_lon ,'latitude': self.scale*df.imag.values + self.center_lat })

        super(AddOnMap, self).__init__(name=name, **kwargs)
        for lat, lon in zip( self.point.latitude, self.point.longitude):
            self.add_child(folium.CircleMarker(location=[lat, lon],
                                               popup=folium.Popup(name, sticky=True),
                                               tooltip=folium.Tooltip(name),
                                               color=color,
                                               radius=.000001,
                                               dash_array=dash_array))

    # Add polyLine
    def add_polyline(self, color, weight, opacity, dash_array, name, **kwargs): 
        locations = list([(lat, lng) for lng, lat in self.point[["longitude","latitude"]].values])

        self.add_child(folium.PolyLine(locations=locations,
                                    color=color,
                                    weight=weight,
                                    opacity=opacity,
                                    dash_array=dash_array,
                                    popup=folium.Popup(name, sticky=True),
                                    tooltip=folium.Tooltip(name),
                                    name=name,
                                    draggable=True,
                                    **kwargs))
    # add circle marker
    def add_circles(self,color, radius, popup, **kwargs):
        for lat, lon in zip( self.point.latitude, self.point.longitude):
            self.add_child(folium.CircleMarker(location=[lat, lon],
                                        popup=folium.Popup(popup, sticky=True),
                                        tooltip=folium.Tooltip(popup),
                                        color=color,
                                        radius=radius,
                                        **kwargs))
   


def get_ellipse_params(df, name):
    """
    Gets ellipse parameters for a given tidal constituent
    """
    
    df = df.loc[name]
    SEMA, SEMI = df['Lsmaj'], df['Lsmin']
    PHA, INC = df['g'], df['theta']
    ECC = SEMI / SEMA
    return SEMA, SEMI, PHA, INC, ECC
    

def tidal_entities(df,consti_name):
    '''
    INPUT:
    df: pandas dataframe of tidal constituent parameters
    consti_name: name of a tidal constituent
    OUTPUT:
    POINTS OF TIDAL ELLIPSE, WMAX, WMIN, CCW, CW
    
    '''
    SEMA, SEMI, PHA, INC, ECC = get_ellipse_params(df,consti_name)

    i = 1j

    SEMI = SEMA * ECC
    Wp = (1 + ECC) / 2 * SEMA
    Wm = (1 - ECC) / 2 * SEMA
    THETAp = INC - PHA
    THETAm = INC + PHA

    # Convert degrees into radians
    THETAp = THETAp / 180 * np.pi
    THETAm = THETAm / 180 * np.pi
    INC = INC / 180 * np.pi
    PHA = PHA / 180 * np.pi

    # Calculate wp and wm.
    wp = Wp * np.exp(i * THETAp)
    wm = Wm * np.exp(i * THETAm)

    dot = np.pi / 36
    ot = np.arange(0, 2 * np.pi, dot)
    a = wp * np.exp(i * ot) 
    b = wm * np.exp(-i * ot) 
    w = a + b

    wmax = SEMA * np.exp(i * INC)
    wmin = SEMI * np.exp(i * (INC + np.pi / 2))

    # df1:tidal ellipse points
    df1 = pd.DataFrame({"real":np.real(w),"imag":np.imag(w)})
    # repeat the first row to close the ellipse
    df1 = df1.append(df1.iloc[0,:], ignore_index=True)
    # df2:tidal ellipse semimajor axis points
    df2 = pd.DataFrame({"real":[0,np.real(wmax)],"imag":[0,np.imag(wmax)]})
    # df3:tidal ellipse semiminor axis points
    df3 = pd.DataFrame({"real":[0,np.real(wmin)],"imag":[0,np.imag(wmin)]})
    df4 = pd.DataFrame({"real":np.real(b),"imag":np.imag(b)})
    # repeat the first row to close the circle
    df4 = df4.append(df4.iloc[0,:], ignore_index=True)
    df5 = pd.DataFrame({"real":np.real(a),"imag":np.imag(a)})
    # repeat the first row to close the circle
    df5 = df5.append(df5.iloc[0,:], ignore_index=True)

    # We don't need the following but just in case

    # df6 = pd.DataFrame({"real":[0,np.real(a[0])],"imag":[0,np.imag(a[0])]})
    # df7 = pd.DataFrame({"real":[0,np.real(b[0])],"imag":[0,np.imag(b[0])]})
    # df8 = pd.DataFrame({"real":[0,np.real(w[0])],"imag":[0,np.imag(w[0])]})
    # df9 = pd.DataFrame({"real":[np.real(a[0])],"imag":[np.imag(a[0])]})
    # df10 = pd.DataFrame({"real":[np.real(b[0])],"imag":[np.imag(b[0])]})
    # df11 = pd.DataFrame({"real":[np.real(w[0])],"imag":[np.imag(w[0])]})
    # df12 = pd.DataFrame({"real":np.real([a[0], a[0]+b[0]]),"imag":np.imag([a[0],a[0]+b[0]])})
    # df13 = pd.DataFrame({"real":np.real([b[0], a[0]+b[0]]),"imag":np.imag([b[0],a[0]+b[0]])})

    # for n in range(len(ot)):
    #     df14 = pd.DataFrame({"real":[np.real(a[n])], "imag":[np.imag(a[n])]})
    #     df15 = pd.DataFrame({"real":[np.real(b[n])], "imag":[np.imag(b[n])]})
    #     df16 = pd.DataFrame({"real":[np.real(w[n])], "imag":[np.imag(w[n])]})

    return df1, df2, df3, df4, df5




def color_selector(n):
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(n)]
    return colors


def add_tidal_entities_on_map(df, consti_name, map_object,center_lat, center_lon, site_name, **kwargs):
    df1, df2, df3, df4, df5 = tidal_entities(df,consti_name)
    colors = color_selector(3) 

    el = AddOnMap(df1, colors[0], '5', f'{site_name}: ellipse',center_lat, center_lon)
    el.add_polyline(colors[0], 2, 1, '5', f'{site_name}: ellipse')
    smj = AddOnMap(df2, colors[0], '5', f'{site_name}: wmax',center_lat, center_lon)
    smj.add_polyline(colors[0], 2, 1, '5', f'{site_name}: wmax')
    smn = AddOnMap(df3, colors[0], '5', f'{site_name} wmin',center_lat, center_lon)
    smn.add_polyline(colors[0], 2, 1, '5', f'{site_name}: wmin')
    cw = AddOnMap(df4, colors[1], '5', f'{site_name}: CW',center_lat, center_lon)
    cw.add_polyline(colors[1], 2, 1, '5', f'{site_name}: CW')
    ccw = AddOnMap(df5, colors[2], '5', f'{site_name}: CCW',center_lat, center_lon)
    ccw.add_polyline(colors[2], 2, 1, '5', f'{site_name}: CCW')

    # el = AddOnMap(df1, 'blue', '5', f'{site_name}: ellipse',center_lat, center_lon)
    # el.add_polyline('blue', 2, 1, '5', f'{site_name}: ellipse')
    # smj = AddOnMap(df2, 'black', '5', f'{site_name}: wmax',center_lat, center_lon)
    # smj.add_polyline('black', 2, 1, '5', f'{site_name}: wmax')
    # smn = AddOnMap(df3, 'black', '5', f'{site_name} wmin',center_lat, center_lon)
    # smn.add_polyline('black', 2, 1, '5', f'{site_name}: wmin')
    # cw = AddOnMap(df4, 'green', '5', f'{site_name}: CW',center_lat, center_lon)
    # cw.add_polyline('green', 2, 1, '5', f'{site_name}: CW')
    # ccw = AddOnMap(df5, 'red', '5', f'{site_name}: CCW',center_lat, center_lon)
    # ccw.add_polyline('red', 2, 1, '5', f'{site_name}: CCW')
    # add everything to map
    [map_object.add_child(i) for i in [el, cw, ccw, smj, smn]]
    return map



def angle_between_smjs(A, B, C, /):
    '''
    To be used for calculating the angle between the semimajor axes of two ellipses
    '''
    Ax, Ay = A[0]-B[0], A[1]-B[1]
    Cx, Cy = C[0]-B[0], C[1]-B[1]
    a = atan2(Ay, Ax)
    c = atan2(Cy, Cx)
    if a < 0: a += pi*2
    if c < 0: c += pi*2
    return (pi*2 + c - a) if a > c else (c - a)

# map.save('/home/zelalem/my_repos/mygithubrepos/adcp-heading-verification/html/foluim_ellipses.html')
# webbrowser.open('file://' + '/home/zelalem/my_repos/mygithubrepos/adcp-heading-verification/html/foluim_ellipses.html')

# if __name__ == '__main__':
#     import sys
#     sys.exit(main(sys.argv))

