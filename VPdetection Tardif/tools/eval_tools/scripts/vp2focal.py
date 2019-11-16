import os
import numpy as np


def vp2focal(vsVP, image_size):
    # vsVP: num_vps x 2, [-1,1], number of VP = 2, 3
    height = image_size[0]
    width = image_size[1]
    M = -np.matrix([[(height / 2.0) ** 2, 0], [0, (width / 2.0) ** 2]])
        
    num_vps = np.array(vsVP).shape[0]
    
    focal_list = []
    for i in range(num_vps-1):
        for j in range(i+1, num_vps):
            vp_i = np.matrix([vsVP[i]])
            vp_j = np.matrix([vsVP[j]])
            focal = float(vp_i * M * vp_j.T)
            if focal > 0:
                focal_list.append(focal)

    if len(focal_list) == 0:
        return 0
    
    focal_proposal = np.stack([-np.array(focal_list), np.ones(len(focal_list))]).T  # attention the -1 here
    [u,d,v] = np.linalg.svd(focal_proposal)  # svd in python is different in maltab, v = v.T
    sol = v[-1]
    focal_sq = sol[1]/ sol[0]
    
    if focal_sq > 0:
        focal = float(np.sqrt(focal_sq))
    else:
        focal = 0.0
   
    return focal


if __name__ == '__main__':
    vps = np.array([[0.01045462743200917, 2.7087946179954203], [250.30290305219467, -0.9467908387676157], [-0.036817415798437024, -1.7657862490269127]])
    image_size = [480, 640]
    
    focal = vp2focal(np.array(vps), image_size)
    print(focal)
