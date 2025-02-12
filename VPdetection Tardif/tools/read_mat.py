import scipy.io as sio
import numpy as np
import json
import os


def load_data(data_name):
    data = sio.loadmat(data_name)
    prediction = data['prediction'][0,0]
    image_path, image_size, line_segs, vps, group  = prediction
    
    image_path = image_path[0]
    image_size = image_size.tolist()
    image_size = [image_size[0][0], image_size[0][1]]  # height x width
  
    line_segs = line_segs.T.reshape(-1, 4).tolist()
    vps = vps.T.tolist()
    group = (group[0].astype(np.int) - 1).tolist()

    vps = vps[:3]
    new_group = []
    for g in group:
        if g < 3:
            new_group.append(g)
        else:
            new_group.append(-1)
    group = new_group

    return image_path, image_size, line_segs, vps, group


def point2line(end_points):
    # line: ax + by + c = 0, in which a^2 + b^2=1, c>0
    # point: 2 x 2  # point x dim
    # A = np.matrix(end_points) - np.array(image_size) / 2
    # result = np.linalg.inv(A) * np.matrix([1,1]).transpose()

    A = np.asmatrix(end_points)
    result = np.linalg.inv(A) * np.asmatrix([-1, -1]).transpose()  # a, b, 1
    a = float(result[0])
    b = float(result[1])
    norm = (a ** 2 + b ** 2) ** 0.5
    result = np.array([a / norm, b / norm, 1 / norm])

    return result


def lineseg2line(line_segs, image_size):
    # line_segs: number x (width, heigth)
    height, width = image_size
    new_line_segs = []
    new_lines = []
    for line_s in line_segs:
        end_points = [[line_s[1], line_s[0]], [line_s[3], line_s[2]]]
        new_line_segs.append(end_points)
        new_end_points = [[(end_points[i][0] - image_size[0] / 2 ) / (image_size[0] / 2),
                            (end_points[i][1] - image_size[1] / 2 ) / (image_size[1] / 2)]
                            for i in range(2)]
        new_line = point2line(new_end_points).tolist()
        new_lines.append(new_line)

    return new_line_segs, new_lines
        

def process(data_list, save_path):
    save_op = open(save_path, 'w')

    for data_name in data_list:
        print(data_name)
        image_path, image_size, line_segs, vps, group = load_data(data_name)
        
        # there are overlap for each group
        # image_size: height x width
        vps_output = []
        norm = max(image_size)
        for vp in vps:
            new_vp = [(vp[1] / vp[2] * norm - image_size[0] / 2) / (image_size[0] / 2), 
                      (vp[0] / vp[2] * norm - image_size[1] / 2) / (image_size[1] / 2)]
            vps_output.append(new_vp)
        
        line_segs_output, new_lines_output = lineseg2line(line_segs, image_size)
        group_output = group

        image_names = image_path.split('/')
        image_name = os.path.join(image_names[-2], image_names[-1])
        
        json_out = {'image_path': image_name, 'line': new_lines_output, 'org_line': line_segs_output, 
                'group': group_output, 'vp': vps_output} 

        json.dump(json_out, save_op)
        save_op.write('\n')


if __name__ == '__main__':
    data_name = 'SUNCG'   # 'YUD', 'ScanNet', 'SceneCityUrban3D', 'SUNCG'

    path = '/n/fs/vl/xg5/workspace/baseline/cis400/VPdetection Tardif/dataset/' + data_name + '/output'
    dir_list = [os.path.join(path, dir_path) for dir_path in os.listdir(path)]
    data_list = []
    for dirs in dir_list:
        data_list += [os.path.join(dirs, dir_path + '/data.mat') for dir_path in os.listdir(dirs)]

    save_path = '/n/fs/vl/xg5/workspace/baseline/cis400/VPdetection Tardif/dataset/' + data_name + '/data'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'data.json')
    process(data_list, save_file)


