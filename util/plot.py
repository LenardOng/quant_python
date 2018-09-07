import numpy as np
from matplotlib import pyplot as plt

# Tableau colours for easier replication, colour values from
# https://public.tableau.com/profile/chris.gerrard#!/vizhome/TableauColors/ColorPaletteswithRGBValues
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    


gray = np.repeat(199/255, 3)  
             
             
def plot_nice_lineplot(x, y, title=None, y_label=None, x_label=None, ax=None, col_index=0, line_label=None, linestyle='-'):
    if not ax:
        ax = plt.subplot(111)

    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
   
    
    plt.plot(x, y, lw=2.5, color=[x/255.0 for x in tableau20[col_index]], linestyle=linestyle)
    ax.grid(color=gray, linestyle='--', linewidth=0.8)
    #for y in range(10, 91, 10):    
    #    plt.plot(range(1968, 2012), [y] * len(range(1968, 2012)), "--", lw=0.5, color="black", alpha=0.3)    
    plt.xticks(fontsize=12)    
    plt.yticks(fontsize=12) 
    if title:
        plt.title(title, fontsize=16)
    if y_label:
        plt.ylabel(y_label)
    if x_label:
        plt.xlabel(x_label)
        
    if line_label:
        y_pos = y[-1]
        plt.text(2016.5, y_pos-1000, line_label, fontsize=14, color=[x/255.0 for x in tableau20[col_index]])  
    return 0




