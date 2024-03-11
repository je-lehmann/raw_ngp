import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np

# in RAD
def cartesian_to_spherical(x, y, z):
    rho = math.sqrt(x**2 + y**2 + z**2) # radial distance, this should nearly be the same for all points
    phi = math.atan2(y, x)  # azimuthal angle
    theta = math.acos(z / rho)  # zenith angle
    return rho, phi, theta

def spherical_to_cartesian(rho, phi, theta):
    x = rho * math.sin(theta) * math.cos(phi)
    y = rho * math.sin(theta) * math.sin(phi)
    z = rho * math.cos(theta)
    return x, y, z

# Provided data
# 11_15 0.343855 1.425504 1.552041
# 12_15 0.341225 1.396592 1.629512
f = open("/graphics/projects2/data/light_stage/calibration/led_positions_white_25.10.2017")
#f = open("led_positions_white_25.10.2017")
measured_z = 1.35

yourList = f.readlines()
numpy_coords = np.array([
    (
        float(line.split()[1]),#-0.4, #offset to center
        float(line.split()[2]),#-0.25,
        float(line.split()[3]),#-1.0
    )
    for line in yourList
])
CM = np.average(numpy_coords, axis=0)
coords = [
    (
        line.split('_')[0],
        float(line.split()[1]) - CM[0],
        float(line.split()[2]) - CM[1],
        float(line.split()[3]) - CM[2],
    )
    for line in yourList
]
print(CM)
direction_vectors = [
    (label + 'r', -x, -y, -z)
    for label, x, y, z in coords
]

spherical_coordinates = []
valid_leds2 = [92,93,94,95,96,97,101,102,103,104,105,110,116,117,118,119,120,121,122,123,124,125,126,127,128,
              129,130,131,132,133,134,146,147,148,149,150,151,152,153,158,160,161,162,163,164,165,166,167] #48
valid_leds = [92,95,96,102,104,110,117,118,121,123,125,127,
              129,131,133,146,148,150,152,158,161,163,165,167] #24

for label, x, y, z in coords:
    theta, phi, rho = cartesian_to_spherical(-x, -y, -z) #light dir points towards origin
    label = label + 'sh'
    spherical_coordinates.append((label, theta, phi, rho))

with open('/home/lehmann/scratch2/captures/spherical_coords_LEDs_RAD', 'w') as f:
    for item in spherical_coordinates:
        item = str(item)
        item = item.replace('(', '')
        item = item.replace(')', '')
        item = item.replace(',', '')
        item = item.replace('\'', '')
        f.write(item + '\n')

debugSH = [] # convert sphericals back for checking
for labels_sh, rho, phi, theta in spherical_coordinates:
    x_sh, y_sh, z_sh = spherical_to_cartesian(rho, phi, theta)
    debugSH.append((label, x_sh, y_sh, z_sh))

# Extracting data for each column
filtered_coords = [(label, x, y, z) for label, x, y, z in coords if int(label) in valid_leds]
labels, xf, yf, zf = zip(*filtered_coords)
labels, x, y, z = zip(*coords)
#labels_r, x_r, y_r, z_r = zip(*direction_vectors)
#labels_sh, x_sh, y_sh, z_sh = zip(*debugSH)

# Creating the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot LED Coords, direction vectors and spherical coords
#ax.scatter(x, y, z, c='b', marker='o')
ax.scatter(xf, yf, zf, c='r', marker='o')

#ax.quiver(x, y, z, x_r, y_r, z_r, color='y', arrow_length_ratio=0.01, linewidth=0.2, alpha=0.4) # start -> dir
#ax.quiver(x, y, z, x_sh, y_sh, z_sh, color='r', arrow_length_ratio=0.01, linewidth=0.2, alpha=0.4) # start -> dir

# Adding labels to each point
for i, label in enumerate(labels):
    ax.text(x[i] + 0.04, y[i], z[i], label, fontsize=8, color='black')

# Adding labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('LightStage LEDs 3D Scatter Plot')

# Display the plot
plt.show()
