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

def spherical_to_cartesian_unit_vector(phi, theta):
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    
    # Normalize to get unit vector
    magnitude = math.sqrt(x**2 + y**2 + z**2)
    x /= magnitude
    y /= magnitude
    z /= magnitude
    
    return x, y, z

def load_light_dirs():
    f = open("/graphics/projects2/data/light_stage/calibration/led_positions_white_25.10.2017")
    measured_z = 1.35

    ldirList = f.readlines()
    numpy_coords = np.array([
        (
            float(line.split()[1]),#-0.4,
            float(line.split()[2]),#-0.25,
            float(line.split()[3]),#-1.0
        )
        for line in ldirList
    ])
    CM = np.average(numpy_coords, axis=0)
    coords = [
        (
            line.split('_')[0],
            float(line.split()[1]) - CM[0],
            float(line.split()[2]) - CM[1],
            float(line.split()[3]) - CM[2],
        )
        for line in ldirList
    ]

    spherical_coordinates = []
    for id, x, y, z in coords:
        rho, phi, theta = cartesian_to_spherical(-x, -y, -z) # light dir points towards origin
        spherical_coordinates.append([phi, theta])
    
    unit_vectors = [spherical_to_cartesian_unit_vector(phi, theta) for phi, theta in spherical_coordinates]
    return unit_vectors
    