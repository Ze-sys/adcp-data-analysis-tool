
import math
import numpy as np
import pandas as pd
from rosely import WindRose

def plot_single_currentrose(t,depth_avg_us,depth_avg_vs, site_name):

        wd =np.array(list(map(math.atan2,depth_avg_us, depth_avg_vs)))
        ws = np.array(np.sqrt(depth_avg_us**2 + depth_avg_vs**2 ))

        df = pd.DataFrame({'date':t,'ws':ws,'wd':wd*180/np.pi+180})

        WR = WindRose(df)
        WR.calc_stats(normed=True, bins=6)

        fg = WR.plot(
            colors='Jet',template='plotly_dark',

            title='<b>current speeds and directions</b>',
            output_type='return',range_r =[0, 40],
            colors_reversed=False,
            height=400, 
            width=400,
            labels={"<b>speed":"current speed (m/s)</b>"},
#             hovertemplate = 'Price: $%{y:.2f}'+'<br>Week: %{x}',
        )
        # update hovertemplate to add site names
        for i in range(len(fg['data'])):

            fg['data'][i].hovertemplate = "{}{}".format(f"site name: {site_name}<br>",fg['data'][i].hovertemplate) 

        fg.update_layout(
                    #   title_text=f"depth averaged current speeds and directions",
                      legend=dict(
                        yanchor="middle",
                        xanchor="left",
                           x=1.1,
                           y=0.75,
                      orientation='v'),
                margin=dict(
                l=20,
                r=0,
                b=20,
                t=80,
                pad = 0
            )


                    )

        return fg