import os
import glob


def run(task_name):
    save_name = '../error_logs/' + task_name + '.txt'
    file_list = glob.glob('../error_logs/' + task_name + '_error_*_error.txt')
    
    with open(save_name, 'w') as w_op:
        for file_name in file_list:
            with open(file_name, 'r') as r_op:
                lines = r_op.readlines()
                for line in lines:
                    print(line)
                    w_op.write(line)


if __name__ == '__main__':
    task_list = ['ScanNet', 'SceneCityUrban3D', 'SUNCG']
    for task_name in task_list:
        run(task_name)
    
