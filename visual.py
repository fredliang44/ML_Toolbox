from bokeh.plotting import figure, show
from bokeh.io import output_notebook

class Line(object):

    # setup plot
    def __init__(self,x=None,y=None,x_label = None,y_label = None,title=None):
        # create a new plot
        self.title = title
        self.x = x
        self.y = y
        self.x_label = x_label
        self.y_label = y_label
        self.p = figure(title = self.title)
    def show(self):
        # add some renderers
        self.p.line(self.x, self.x, legend=self.x_label)
        self.p.circle(self.x, self.x,legend=self.x_label, fill_color="white", size=8)
        # self.p.line(self.x, self.y, legend=self.y_label, line_width=3)
        self.p.line(self.x,self.y, legend=self.y_label, line_color="red")
        self.p.circle(self.x,self.y, legend=self.y_label, fill_color="red", line_color="red", size=6)
        # self.p.line(self.x, self.y2, legend="y=10^x^2", line_color="orange", line_dash="4 4")
        output_notebook()
        show(self.p)



class Dot(object):

    # setup plot
    def __init__(self,x1=None,y1=None,x2=None,y2=None,x_label = None,y_label = None,title=None):
        # create a new plot
        self.title = title
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x_label = x_label
        self.y_label = y_label
        self.p = figure(title = self.title)
    def show(self):
        # add some renderers
        self.p.circle(self.x1, self.y1, color="blue", fill_alpha=0.2, size=10)
        self.p.circle(self.x2, self.y2, color="red", fill_alpha=0.2, size=10)
        output_notebook()

        show(self.p)
