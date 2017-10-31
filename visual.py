from bokeh.plotting import figure, show
from bokeh.io import output_notebook

class Line(object):
    def __init__(self, x, y,y1,y2):
        # create a new plot
        self.x = x
        self.y = y
        self.y1 = y1
        self.y2 = y2
        self.p = figure(
            tools="pan,box_zoom,reset,save",
            y_axis_type="log", y_range=[0.001, 10 ** 11], title="log axis example",
            x_axis_label='sections', y_axis_label='particles'
        )

    def show(self):
        # add some renderers
        self.p.line(self.x, self.x, legend="y=x")
        self.p.circle(self.x, self.x, legend="y=x", fill_color="white", size=8)
        self.p.line(self.x, self.y, legend="y=x^2", line_width=3)
        self.p.line(self.x,self.y1, legend="y=10^x", line_color="red")
        self.p.circle(self.x,self.y1, legend="y=10^x", fill_color="red", line_color="red", size=6)
        self.p.line(self.x, self.y2, legend="y=10^x^2", line_color="orange", line_dash="4 4")
        output_notebook()
        show(self.p)




