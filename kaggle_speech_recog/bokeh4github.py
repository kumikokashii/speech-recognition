# For github, that doesn't show bokeh graphs

from bokeh.plotting import show
import builtins
import IPython.display as ipd
from bokeh.io import export_png
import os

use_bokeh_screenshot = False
if hasattr(builtins, 'use_bokeh_screenshot'):
    use_bokeh_screenshot = builtins.use_bokeh_screenshot

if use_bokeh_screenshot:
    def show(p):
        filename = 'temp_img.png'
        ipd.display(ipd.Image(export_png(p, filename=filename)))
        os.remove(filename)
