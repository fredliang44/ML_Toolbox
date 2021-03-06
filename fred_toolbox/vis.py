from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import HoverTool, BoxSelectTool, BoxZoomTool, ResetTool, LassoSelectTool, SaveTool


class Line(object):

    # setup plot
    def __init__(self, x=None, y=None):
        # create a new plot
        self.title = None
        self.x = x
        self.y = y
        self.label1 = None
        self.label2 = None
        self.x_label = "x_label"
        self.y_label = "y_label"
        self.tools = [BoxZoomTool(), ResetTool(), BoxSelectTool(), LassoSelectTool(), SaveTool(), HoverTool(
            tooltips=[
                ("index", "$index"),
                ("(" + self.x_label + "," + self.y_label + "y)", "($x, $y)"),
            ])]

    def show(self):
        self.p = figure(x_axis_label=self.x_label, y_axis_label=self.y_label, tools=self.tools)
        self.p.title.text = self.title
        self.p.title.text_font_size = "25px"
        self.p.line(self.x, self.x, legend=self.label1)
        self.p.circle(self.x, self.x, legend=self.label1, fill_color="white", size=6)
        self.p.line(self.x, self.y, legend=self.label2, line_color="red")
        self.p.circle(self.x, self.y, legend=self.label2,
                      fill_color="red", line_color="red", size=6)
        output_notebook()
        show(self.p)


class Dot(object):

    # setup plot
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        # create a new plot
        self.title = None
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x_label = "x_label"
        self.y_label = "y_label"
        self.label1 = None
        self.label2 = None

    def show(self):
        self.p = figure(x_axis_label=self.x_label, y_axis_label=self.y_label, tools=self.tools)
        self.p.title.text = self.title
        self.p.title.align = "center"
        self.p.title.text_font_size = "25px"
        self.p.circle(self.x1, self.y1, legend=self.label1, color="blue", fill_alpha=0.2, size=10)
        self.p.circle(self.x2, self.y2, legend=self.label2, color="red", fill_alpha=0.2, size=10)
        output_notebook()
        show(self.p)
