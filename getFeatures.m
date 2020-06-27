function [ ] = getBlobss( data_path )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% first we locate 'data/HKUST/' 
% then we get all classes using dir()
% then we iterate through all classes using for 
% the corresponding blob file is located in the blobs sub-directory
% we make a target t = {}; t.config=configure(); t.image= imread(); t.blob_image = imread()
% make it go through all the things in bin_features
% we have to calculate the derived features
% then save this to a matrix 
% go to next image
% after we go through all images in a class, save the csv in root class dir as <class>_feat.csv eg. Akahiwo_feat.csv
% go to next class
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
        disp([ii '/' length(images)]);
        t = {};
        t.config = configure();
        t.image = imread([data_path classDir(i).name filesep images(ii).name]);
        t.blob_image = imread([data_path classDir(i).name filesep 'blob/' images(ii).name]);

        t = blob_geomprop(t);
        t = blob_rotate(t);
        t = blob_texture(t);
        t = blob_invmoments(t);
        t = blob_shapehist_stats(t);
        t = blob_RingWedge(t);
        t = biovolume(t);
        t = blob_sumprops(t);
        t = blob_Hausforff_symmetry(t);
        t = image_HOG(t);
        t = blob_rotated_geomprop(t);

        disp("hi");
    end
end

end

