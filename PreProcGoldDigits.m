% CLEAR ALL
clearvars; close all; clc;

% SET UP
addpath(genpath(pwd));
i=1;
currentFolder ='Datasets\GoldStandardDigits';
images = dir(fullfile(currentFolder,'*.jpg'));
file = fullfile(currentFolder, images(i).name);
image = imread(file);

fprintf('%s', file)

imgGray = rgb2gray(image); 
imgGray = histeq(imgGray); 

% Binarize and invert image
mask = imbinarize(imgGray,0.52);
mask = ~mask;

subplot(3,3,7);
imshow(mask);
title('Isolated Black Pixels');

imwrite(mask, 'C:\Users\ginan\OneDrive\Documents\Semester 2\Embedded Image Processing\MiniProject\Datasets\GoldStandardDigits\inv80.jpg', 'jpg')