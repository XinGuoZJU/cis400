clear
close all

datapath = '/n/fs/vl/xg5/Datasets/YUD/YorkUrbanDB';
savepath = 'dataset/YUD/output';

dirs = dir(datapath);

for i = 3:size(dirs,1)
    dir_name = dirs(i).name;
    dirpath = [datapath, '/', dir_name];
    if isdir(dirpath)
        image_name = [dirpath, '/', dir_name, '.jpg'];
        save_path = [savepath, '/', dir_name];
        run(image_name, save_path);
    end
end
