clear
close all

dataset_name = 'SceneCityUrban3D';

if strcmp(dataset_name, 'YUD')
    datapath = '/n/fs/vl/xg5/Datasets/YUD/YorkUrbanDB';
    savepath = 'dataset/YUD/output';
    img_type = 'jpg';
elseif strcmp(dataset_name, 'ScanNet')
    datapath = '/n/fs/vl/xg5/Datasets/ScanNet/scannet-vp';
    savepath = 'dataset/ScanNet/output';
    img_type = 'png';
elseif strcmp(dataset_name, 'SceneCityUrban3D')
    datapath = '/n/fs/vl/xg5/Datasets/SceneCityUrban3D/su3';
    savepath = 'dataset/SceneCityUrban3D/output';
    img_type = 'png';
elseif strcmp(dataset_name, 'SUNCG')
    datapath = '/n/fs/vl/xg5/Datasets/SUNCG/mlt_v2';
    savepath = 'dataset/SUNCG/output';
    img_type = 'png';
end


dirs = dir(datapath); % struct
for i = 3:size(dirs,1)
    dir_name = dirs(i).name;
    dirpath = [datapath, '/', dir_name];
    if isdir(dirpath)
        image_list = dir([dirpath, '/*.', img_type]); % struct
        for j = 1: size(image_list,1)
            img_name = image_list(j).name;
            image_name = [dirpath, '/', img_name];

            save_img_dir = strsplit(img_name, '.'); % cell
            save_path = [savepath, '/', dir_name, '/', save_img_dir{1}];
            try
                run(image_name, save_path);
            catch
                fileID = fopen([dataset_name, '_error.txt'], 'a');
                fprintf(fileID, [image_name, '\n']);
                fclose(fileID);
            end
        end
    end
end
