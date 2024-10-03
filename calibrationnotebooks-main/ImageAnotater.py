# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib widget
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display
import ipywidgets as widgets
import numpy as np  # For handling images as arrays

import pprint
import dataclasses


# %%
@dataclasses.dataclass
class BoundingBox():
    xmin: int
    ymin: int
    heigth: int
    width: int

    def __repr__(self):
        return f" xmin: {self.xmin: >5}\n ymin: {self.ymin : >5}\n widht:{self.width :>5}\n heigth:{self.heigth : >4}\n" 



# %%
BoundingBox(xmin=1,
            ymin=1,
            width=2,heigth=2)


# %%

# %%
class ImageAnnotator:
    def __init__(self, image_files):
        self.image_files = image_files
        self.current_image_index = 0
        self.bounding_boxes = []
        self.current_boxes = []
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid = None
        self.rect = None
        self.x0, self.y0 = None, None
        self.display_image()
        self.output = widgets.Output()
        
    def display_image(self):
        """Displays the current image."""
        image_path = self.image_files[self.current_image_index]
        self.image = plt.imread(image_path)
        self.ax.clear()
        self.ax.imshow(self.image)
        self.current_boxes = []
        plt.show()
        
    def on_click(self, event):
        """Event handler for mouse click."""
        # Record the start point
        self.x0, self.y0 = event.xdata, event.ydata
        self.rect = patches.Rectangle((self.x0, self.y0), 0, 0, linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(self.rect)
        
    def on_release(self, event):
        """Event handler for mouse release."""
        # Record the bounding box
        x1, y1 = event.xdata, event.ydata
        self.current_boxes.append(BoundingBox(xmin=self.x0,
                                              ymin=self.y0,
                                              width=x1 - self.x0,
                                              heigth=y1 - self.y0))
        self.bounding_boxes.append(self.current_boxes[-1])
        self.rect.set_width(x1 - self.x0)
        self.rect.set_height(y1 - self.y0)
        self.fig.canvas.draw()


    def on_motion(self, event):
        """Event handler for mouse motion. Updates the size of the rectangle."""
        if self.rect is None or event.inaxes != self.ax:
            return  # Ignore motion outside the axes or if no rectangle is started
        x1, y1 = event.xdata, event.ydata
        self.rect.set_width(x1 - self.x0)
        self.rect.set_height(y1 - self.y0)
        self.fig.canvas.draw()
        
    def next_image(self):
        """Loads the next image."""
        if self.current_image_index + 1 < len(self.image_files):
            self.current_image_index += 1
            self.display_image()
        else:
            self.close_annotation()
            display(self.output)


    def close_annotation(self):
        """Clears and closes the figure, and hides the button."""
        self.ax.clear()  # Clear the axes
        plt.close(self.fig)  # Close the figure to release resources
        self.button.layout.display = 'none'  # Hide the button
        with self.output:
            print("Annotation completed. No more images.")
     


# %% [raw]
# image_files = ["test_data/extracted_15.jpg","test_data/extracted_15.jpg"]  # List your images here
# annotator = ImageAnnotator(image_files)
#
# button = widgets.Button(description="Next Image")
# button.on_click(lambda b: annotator.next_image())
# annotator.button = button  # Pass the button to the annotator
# display(button)

# %% [raw]
# for box, im in zip(annotator.bounding_boxes, annotator.image_files):
#     print(im)
#     print(box)
#     print("\n")

# %% [raw]
# test_annotator()

# %%

# %%
