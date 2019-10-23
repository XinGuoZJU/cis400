clear
close all

imgStr = 'demo_data/imgs/P1020171.jpg';
savepath = 'demo_data/output/P1020171';

%includes
addpath(genpath('JLinkage'));
addpath 'lineSegDetect/'

ARGS = FACADE_ARGS_default();
const = FACADE_const();
ARGS.mKinv = [];

%arguments for Vanishing point detection
ARGS.plot = 1; 
ARGS.savePlot = false;

ARGS.manhattanVP = true;
%ARGS.manhattanVP = false;
ARGS.REF_remNonManhantan = true;
ARGS.ALL_calibrated = false;
ARGS.ERROR = const.ERROR_DIST;
load cameraParameters.mat
focal = focal / pixelSize;

%RES.focal = focal;
FCprintf('Ground truth Focal length is %f\n', focal);
%ARGS.mK = [[focal,0,pp(1)];[0,focal,pp(2)];[0,0,1]];
%ARGS.mKinv = inv(ARGS.mK);
%ARGS.imgS = norm([imgW/2,imgH/2]);
%ARGS.imgS = focal;
ARGS.imgStr = imgStr;
%------------------------------------------------------------------------------
%                                Edges and VP
%------------------------------------------------------------------------------
%get data (edges)
%read image
im = imread(imgStr);
im = rgb2gray(im);
image_size = size(im);
if ARGS.plot
  ARGS.savePath = savepath;
  f1 = sfigure(1); 
  clf;
  f = figure('visible','off');
  image(im), colormap(gray(256));
  if ARGS.savePlot
    saveas(f, [ARGS.savePath, '/gray.png']);
  end
end
ARGS.imgS = max(size(im));

%getting edges
[vsEdges,ARGS.imE] = FACADE_getEdgelets2(im, ARGS);

%get vp
ARGS.JL_ALGO=2;
ARGS.JL_solveVP  = const.SOLVE_VP_MAX;
[vsVP,vClass] = FACADE_getVP_JLinkage(vsEdges, im, ARGS);
%vsVP = vsVP(1:3);

vbOutliers = FACADE_getOutliers(ARGS,vsEdges, vClass, vsVP);          

[vsVP, vClass] = FACADE_orderVP_Mahattan(vsVP, vsEdges, vClass);
%[vsVP, vClass] = FACADE_orderVP_nbPts(vsVP, vsEdges, vClass);
[f123, f12] = FACADE_selfCalib(ARGS,vsVP, vsEdges, vClass, vbOutliers);

if ARGS.plot 
  %ploting
  if ARGS.savePlot
    FACADE_plotSegmentation(im,  vsEdges,  vClass, [], vbOutliers, ARGS.savePath); %save
  else
    FACADE_plotSegmentation(im,  vsEdges,  vClass, [], vbOutliers); %-1->don't save
  end
end


%%vanishing point in image space are:
%VPimage = mToUh(ARGS.mK*[vsVP.VP]);

%%this will plot the vanishing points that are inside the image
%sfigure(1);
%hold on
%plot(VPimage(1,1:3), VPimage(2,1:3), '*', 'MarkerSize', 20, 'Color', [1,1,0]);
%hold off


%% vsEdges
% vPts_un: points on line segment
% vPts: normalized points on line segment: the normalized way is divided by max(width, height)
% vL: ax + by + c = 0, (a, b, c) for vPts
vp_len = length(vsEdges);
endpoints = [];
for i = 1:vp_len
    points = vsEdges(i).vPts_un;
    endpoint = [points(:, 1), points(:, end)];
    endpoints = [endpoints, endpoint];
end

vp_list = [];
vp_num = size(vsVP, 2);
for i = 1:vp_num
    vp = vsVP(i).VP;
    vp_list = [vp_list, vp];
end 

prediction.image_path = imgStr;
prediction.image_size = image_size;
prediction.lines = endpoints;   % 2 x (2xnum_lines): two endpoints
prediction.vp = vp_list;
prediction.group = vClass;
save_path = [savepath, '/data.mat'];
save(save_path, 'prediction')
