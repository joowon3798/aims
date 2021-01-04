# import packages
import numpy as np
import hyperspy.api as hs

# load file
sp=hs.load('/home/sychoi/ML-DL-for-TEM-Image-Analysis/MoS2/dm4/Analog_80kV_2576.dm4')

# Read the axis information
# Print all the calibration detail
sp.plot()
sp.metadata



