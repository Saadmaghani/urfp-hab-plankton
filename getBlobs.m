function [ ] = getBlobss( data_path )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% first we locate 'data/HKUST/' 
% then we get all classes using dir()
% then we iterate through all classes using for 
% get all files in the class
% get a target t = {}; t.config=configure(); t.image= imread(); t = blob(t)
% then we save t.blob_image in same class folder but different sub dir

% assumed we call from urfp-hab-plankton folder

classDir = dir(data_path);
classDir = classDir(~ismember({classDir.name},{'.','..'}));
disp(classDir);

for i = 1:length(classDir)
    imagePath = [data_path classDir(i).name filesep];
    images = dir([imagePath '*.png']);
    disp(classDir(i).name);
    blobPath = [data_path classDir(i).name filesep 'blob/'];
    if ~exist(blobPath, 'dir')
        mkdir(blobPath)
    end
    for ii = 1:length(images)
        t = {};
        t.config = configure();
        t.image = imread([data_path classDir(i).name filesep images(ii).name]);
        disp(class(t.image));
        t = blob(t);
        imwrite(t.blob_image, [blobPath images(ii).name]);
    end
end

end

