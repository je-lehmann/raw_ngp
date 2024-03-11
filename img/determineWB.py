import sys, os, glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# path = '/home/lehmann/code/data/colorchecker_cropped.jpg'
def determineWBMat():
    path = '/home/lehmann/code/data/colorchecker_cropped.jpg'
    image = imageio.imread(path)

    # Given b, coordinate array to be used further on to fill in cam matrix.reference colors!
    b = np.array( [[115, 82,  68]
                ,[194, 150, 130]
                ,[98,  122, 157]
                ,[87,  108, 67]
                ,[133, 128, 177]
                ,[103, 189, 170]
                ,[214, 126, 44]
                ,[80,  91,  166]
                ,[193, 90,  99]
                ,[94,  60,  108]
                ,[157, 188, 64]
                ,[224, 163, 46]
                ,[56,  61,  150]
                ,[70,  148, 73]
                ,[175, 54,  60]
                ,[231, 199, 31]
                ,[187, 86,  149]
                ,[8,   133, 161]
                ,[243, 243, 242]
                ,[200, 200, 200]
                ,[160, 160, 160]
                ,[122, 122, 121]
                ,[85,  85,  85]
                ,[52,  52,  52]] ,dtype=float)

    print("b shape", b.shape)
    b_t = np.transpose(b)
    print("b_t shape", b_t.shape)
    # print(b)
    b = (b / 255).astype(float)
    # print(b)
    print("image.shape:", image.shape)
    # Upper left and bottom right coordinates of the first patch.
    coords = np.array([[60,  50],
                    [140, 130]])
    print("coords shape:", coords.shape)
    # Delta, i.e. spacing for the patches in x and y directions.
    delta = 150.0

    # Color for each patch. Default dtype is float32.
    cam = np.zeros([24,3])
    print("cam shape:", cam.shape)
    cam_t = np.transpose(cam)
    print("cam_t shape:", cam_t.shape)

    # fill cam matrix

    counter_patch = 0
    y1 = 50
    y2 = 130
    while coords[0,0] < 630:
        while coords[0,1] < 900:
            sum = [0, 0, 0]
            counter_sum = 0
            for x in range(coords[0, 0], min(630, coords[1, 0]), 1):
                for y in range(coords[0, 1], min(900, coords[1, 1]), 1):
                    sum = [sum[z] + image[x, y, z] for z in range(len(sum))]
                    counter_sum += 1      
            sum[:] = [x / counter_sum for x in sum]
            cam[counter_patch, :] = sum
            counter_patch += 1
            coords[0, 1] += int(delta)
            coords[1, 1] += int(delta)
        coords[0, 0] += int(delta)
        coords[1, 0] += int(delta)
        coords[0, 1] = y1
        coords[1, 1] = y2    

    #print(cam)

    # b) TODO: O = C*WB. We can use the last formula from slide 70 here
    id = np.eye(3)
    mat = np.linalg.solve(np.dot(cam_t, cam), id)
    mat = np.matmul(mat, cam_t)
    mat = np.matmul (mat, b)
    mat = np.transpose(mat)
    #print(mat)
    return mat

wb_mat = np.array([[ 0.0092145, -0.00156694, -0.00049469],
        [-0.00251458, 0.00800111, -0.00052914],
        [ 0.00078036, -0.0038395, 0.0090995 ]])

def applyWB(image):
    wb_mat = determineWBMat()
    image_white_balanced = np.dot(image.astype(float), wb_mat.T) * 255.0
    imageio.imwrite('test.jpg', np.array(image_white_balanced).astype(np.uint8))

image = imageio.imread('./data/wbtest.jpg')
applyWB(image)



