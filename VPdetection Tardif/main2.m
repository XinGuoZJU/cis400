function main2(dataset_name, idx)

%dataset_name = 'YUD';
%idx = 0;

tic
if strcmp(dataset_name, 'YUD')
    index_file = ['/n/fs/vl/xg5/Datasets/YUD/label/index_', num2str(idx), '.txt'];
    datapath = '/n/fs/vl/xg5/Datasets/YUD/YorkUrbanDB';
    savepath = 'dataset/YUD/output';
    img_type = 'jpg';
elseif strcmp(dataset_name, 'ScanNet')
    index_file = ['/n/fs/vl/xg5/Datasets/ScanNet/label/index_', num2str(idx), '.txt'];
    datapath = '/n/fs/vl/xg5/Datasets/ScanNet/scannet-vp';
    savepath = 'dataset/ScanNet/output';
    img_type = 'png';
elseif strcmp(dataset_name, 'SceneCityUrban3D')
    index_file = ['/n/fs/vl/xg5/Datasets/SceneCityUrban3D/label/index_', num2str(idx), '.txt'];
    datapath = '/n/fs/vl/xg5/Datasets/SceneCityUrban3D/su3';
    savepath = 'dataset/SceneCityUrban3D/output';
    img_type = 'png';
elseif strcmp(dataset_name, 'SUNCG')
    index_file = ['/n/fs/vl/xg5/Datasets/SUNCG/label/index_', num2str(idx), '.txt'];
    datapath = '/n/fs/vl/xg5/Datasets/SUNCG/mlt_v2';
    savepath = 'dataset/SUNCG/output';
    img_type = 'png';
elseif strcmp(dataset_name, 'ScanNet_error')
    index_file = ['/n/fs/vl/xg5/workspace/baseline/cis400/VPdetection Tardif/tools/error_case/ScanNet_', num2str(idx), '.txt'];
    datapath = '/n/fs/vl/xg5/Datasets/ScanNet/scannet-vp';
    savepath = 'dataset/ScanNet/output';
    img_type = 'png';
elseif strcmp(dataset_name, 'SceneCityUrban3D_error')
    index_file = ['/n/fs/vl/xg5/workspace/baseline/cis400/VPdetection Tardif/tools/error_case/SceneCityUrban3D_', num2str(idx), '.txt'];
    datapath = '/n/fs/vl/xg5/Datasets/SceneCityUrban3D/su3';
    savepath = 'dataset/SceneCityUrban3D/output';
    img_type = 'png';
elseif strcmp(dataset_name, 'SUNCG_error')
    index_file = ['/n/fs/vl/xg5/workspace/baseline/cis400/VPdetection Tardif/tools/error_case/SUNCG_', num2str(idx), '.txt'];
    datapath = '/n/fs/vl/xg5/Datasets/SUNCG/mlt_v2';
    savepath = 'dataset/SUNCG/output';
    img_type = 'png';
end


fpn = fopen(index_file, 'r');
while ~feof(fpn)
    line_str = fgetl(fpn);
    str_list = strsplit(line_str);
    img_name = str_list{1};
    save_list = strsplit(line_str, '.');
    save_dir = save_list{1};

    image_name = [datapath, '/', img_name];
    save_path = [savepath, '/', save_dir];
    try
        run(image_name, save_path);
    catch
        fileID = fopen([dataset_name, '_', num2str(idx), '_error.txt'], 'a');
        fprintf(fileID, [image_name, '\n']);
        fclose(fileID);
    end
end
fclose(fpn);

toc
