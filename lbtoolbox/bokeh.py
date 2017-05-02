import bokeh
import numpy as np


# output_notebook(resources=bokeh.resources.INLINE)


palette_gg = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']
palette = palette_gg


def style_figure(fig, objects=True):
    """
    Applies my (ggplot2 inspired) style to a figure.

    `objects` are things added to a figure later on, such as the legend.
    These can only be styled if present, meaning this should be called after
    they were added.
    """

    # Change the grid to be gray/white
    fig.background_fill_color= "#E5E5E5"
    fig.grid.grid_line_color="white"
    fig.grid.grid_line_width=0.5

    # change just some things about the x-axes
    fig.axis.axis_line_width = 1
    fig.axis.axis_line_color = 'white'

    # change just some things about the y-axes
    #p.yaxis.major_label_text_color = "orange"
    #p.yaxis.major_label_orientation = "vertical"

    fig.axis.major_tick_in = -1
    fig.axis.major_tick_out = 5
    fig.axis.major_tick_line_width = 1
    fig.axis.major_tick_line_color = '#555555'
    fig.axis.major_label_text_color = '#555555'

    fig.axis.minor_tick_in = -1
    fig.axis.minor_tick_out = 3
    fig.axis.minor_tick_line_width = 0.5
    fig.axis.minor_tick_line_color = '#555555'

    fig.axis.axis_label_text_color = '#555555'

    #fig.title.align = 'center'  # Doesn't work well yet.

    #ct = fig.tools[0]
    #ct.line_color = '#555555'
    #ct.line_width = 2

    if objects:
        fig = style_figure_objects(fig)

    return fig


def style_figure_objects(fig):
    fig.legend.background_fill_alpha = 0.33
    fig.legend.padding = 5  # Space inside legend
    #fig.legend.margin = 0  # From legend to figure border

    return fig


def showdistr(values, x=None, percentiles=[0, 25, 50, 75, 100], name=None, colors=None):
    p = bokeh.plotting.figure(plot_width=450, plot_height=250, responsive=True, tools='pan,wheel_zoom,box_zoom,reset')

    if name is not None:
        p.yaxis.axis_label = "{} percentiles".format(name)
        p.xaxis.axis_label = "Time"

    if x is None: x = np.arange(len(values))
    if colors is None:
        # Use blues but make it symmetric.
        colors = bokeh.palettes.brewer['Blues'][(len(percentiles)+1)//2]
        colors = colors[::-1] + colors[len(percentiles) % 2:]

    qvals = np.nanpercentile(values, percentiles, axis=1)
    for q, qvs, c in zip(percentiles, qvals, colors):
        p.line(x, qvs, color=c, legend='{:.0%}-ile'.format(q/100))

    return p
