% CLEAR ALL
clearvars; close all; clc;

% SET UP
addpath(genpath(pwd));
i=1;
currentFolder ='Datasets\GoldStandardDigitsResize';
images = dir(fullfile(currentFolder,'*.jpg'));
for i = 1:length(images)
    file = fullfile(currentFolder, images(i).name);
    image = imread(file);
    %image = imresize(image,[170 130]);
    %image = 2rgb(image);
    %loc = ['C:\Users\ginan\OneDrive\Documents\Semester 2\Embedded Image Processing\MiniProject\Datasets\GoldStandardDigits\' filename]
    imwrite(image, file, 'png')
end