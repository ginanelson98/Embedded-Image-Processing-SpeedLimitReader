% =========================================================================
% EE551 Mini-Project
% Detection and Classification of Speed Limit Signs
% =========================================================================
% GEORGINA NELSON - 16332886
% =========================================================================
% 3 Datasts Provided:
% - Development set (5 classes)
%		- 20kph
%		- 30kph
%		- 50kph
%		- 80kph
%		- 100kph
% - Gold Standard set
% - Stress set
% =========================================================================
% Programme flows as follows:
% 	1. Sign Detection
%       a. Enhancements
% 		b. Circle Detection
%   2. Object Extraction
%       a.Preprocessing for Digit Isolation
%		b. Digit Isolation
%   3. Object Classification
%		a. Digit Classification
%		b. Speed Limit Extraction
% Each phase represented by MatLab method
% =========================================================================

% CLEAR ALL
clearvars; close all; clc;

% SET UP
addpath(genpath(pwd));

% DATA
%dataset = '30kph';
dataset = 'StressDataset';

% MAIN SYSTEM CALL
ClassifyDataset(dataset);  
                           
% =========================================================================
% DETECT AND CLASSIFY SPEED LIMIT IMAGES
% Takes in folder to be analysed by system
% Calls other methods to carry out system functionality
% Calculates accuracies, presents results and generates system performance
% data
function ClassifyDataset(dataset)
    % INITIALIZE VALUES
    totalConfidence = 0;
    correctClassifications = 0;
    
    % DATA
    if (dataset == "StressDataset")
        loaction = strcat('Datasets/', dataset);
        files = dir(fullfile(loaction,'*.TIF'));
        imgCount = length(files);
    else
        loaction = strcat('Datasets/Development/', dataset);
        files = dir(fullfile(loaction,'*.jpg'));
        imgCount = length(files);
        trueSpeedLimit = str2double(dataset(1:end-3)); % drop kph
    end
    
    % LOOP OVER IMAGES
    for i = 1:imgCount
        % G.TRUTH AND IMAGE PREPARATION
        if (dataset == "StressDataset")
            trueSpeedLimit = str2double(files(i).name(2:3));
        end
        fileName = fullfile(loaction, files(i).name);
        image = imread(fileName);
        
        % =================================================================
        % PHASE 1: SIGN DETECTION
        sign = DetectSign(image);
        
        % PHASE 2: DIGIT EXTRACTION
        digit = ExtractDigit(sign);    
        
        % PHASE 3: OBJECT CLASSSIFICAITON
        [speedLimit, confidence] = ClassifyObject(digit);
        
        % =================================================================
        % OUTPUT
        subplot(1,2,1);
        imshow(image);
        title('Original Image');
        drawnow;
        
        file = [num2str(speedLimit) '.jpg'];
        fileName = fullfile('GoldStandard', file);
        goldStandardImage = imread(fileName);
        subplot(1,2,2);
        imshow(goldStandardImage);
        title('Classified as');
        drawnow;
        
        % =================================================================
        % RESULT AGGRAGATION
        if speedLimit == trueSpeedLimit
            correctClassifications = correctClassifications + 1;   
            result = ' Correct';
        else
            result = ' Incorrect';
        end
        totalConfidence = totalConfidence + confidence;

        % PRINT RESULTS
        fprintf('\nImage %d : True Speed Limit = %d     Assigned = %d     With confidence: %.2f%%%', i, trueSpeedLimit, speedLimit, confidence);
        fprintf('  -> Result = %s ', result);
    end

    % PERFORMANCE EVALUTAION
    systemAccuracy = correctClassifications/imgCount*100;
    averageConfidence = totalConfidence/imgCount;
    
    % PRINT PERFORAMCE EVALUATION
    fprintf('\nSystem Performance Evaluation:\n');
    fprintf('Accuracy: %.2f%% (%d/%d)\n', systemAccuracy, correctClassifications, imgCount);
    fprintf('Average Confidence: %.2f%%%', averageConfidence);

end

% =========================================================================
% PHASE 1 : SIGN DETECTION
% Takes in image to be processed
% Calls 2 main steps:
%   1. Enhance image
%   2. Detect red circle
% Returns detected sign area of interest
function sign = DetectSign(image)  
    enhancedImage = Enhance(image);
    sign = DetectCircle(enhancedImage, image);
end

% =========================================================================
% IMAGE ENHANCEMENT FOR CIRCLE DETECTION
% Takes in image to be enhanced
% Enhancements include:
% Brightening, saturating, sharpening, resizing and isolating red pixels
% Returns enhanced image
function enhancedImage = Enhance(image)
    % INITIALIZE
    dims = [450 450];
    briThresh = 80;
    satThresh = 0.3;
    Y = 0.005;
    minHueThresh = 300/360;
    maxHueThresh = 60/360;   
    minSatThresh = 0.45;
    maxSatThresh = 1.0;
    
    % =====================================================================
    % RESIZE IMAGE
    image = imresize(image,dims);
    
    % =====================================================================
    % BRIGHNESS ENHANCEMENT
    
    % GREYSCALE CONVERSION TO GET AVG IMAGE BRIGHTNESS
    greyscaleImage = rgb2gray(image);
    avgBrightness = mean(greyscaleImage(:));
    
    % RELATIVE BRIGHTNESS ENHANCEMENT
    if avgBrightness < briThresh
        image = imadjust(image, [], [], avgBrightness * Y);
    end
    
    % =====================================================================
    % SHARPNESS ENHANCEMENT
    image = imsharpen(image,'Radius',2,'Amount',1);   
    
    % =====================================================================
    % SATURATION ENHANCEMENT
    
    % CONVERT TO HSV IMAGE AND ISOLATE CHANNELS
    imageHSV = rgb2hsv(image);
    % HUE, SATURATION & VALUE
    h = imageHSV(:, :, 1);
    s = imageHSV(:, :, 2);
    v = imageHSV(:, :, 3);

    % IF BELOW THRESHOLD: INCREASE IMAGE SATURATION
    if s < satThresh
        s = s * 3.5;
    end
        
    % =====================================================================
    % RED PIXEL ISOLATION   
    enhancedImage = (s >= minSatThresh & s <= maxSatThresh) & (h >= minHueThresh | h <= maxHueThresh);
end

% =========================================================================
% CIRCLE DETECTION
% Takes in original and enhanced image from previous step
% Creates bouding box for enhanced circle of interest
% Returns cropped sign from original image
function sign = DetectCircle(enhancedImage, image)  
    % INITIALIZE
    minPixelAreaThresh = 20;
    radThresh = 140;
    rEst =150;
    cEst = [240,240];

    % NOISE REDUCTION
    enhancedImage = bwareaopen(enhancedImage, minPixelAreaThresh);

    % CHARACTERISTICS OF AREAS OF INTEREST & NOISE REDUCTION
    properties = regionprops('table', enhancedImage, 'Centroid', 'Area', 'EquivDiameter'); 
    properties((properties.Area < (minPixelAreaThresh*10)),:) = [];
    diameters = properties.EquivDiameter;
    radii = diameters/2;

    % INCREASE RADIUS IF TOO SMALL
    if radii < radThresh
        radii = radii*1.1;
    end

    % PROPERTIES OF MAIN REGION OF INTEREST
    properties = sortrows(properties, 1, 'descend');
    if (height(properties) > 1)
        r = max(radii);
        c = properties.Centroid(1,:);
    elseif (height(properties) == 1)
        r = radii;
        c = properties.Centroid(1,:);
    else
        r = rEst;
        c = cEst;
    end

    % CENTROID X & Y COORDS
    cXCoord = c(:,1);
    cYCoord = c(:,2);

    % BOUDING BOX X & Y COORDS
    x1 = abs(ceil(cXCoord - r));
    x2 = round(cXCoord + r);
    y1 = abs(ceil(cYCoord - r));
    y2 = round(cYCoord + r);

    % ENSURE BOUNDING BOX REGION STAYS INSIDE IMAGE BOUNDARIES
    [h, w, ~] = size(image);
    if x1 <= 0, x1 = 1;   end
    if y1 <= 0, y1 = 1;   end
    if x2 >= w, x2 = w-1; end
    if y2 >= w, y2 = w-1; end

    if x1 > w || x2 > w || y1 > h || y2 > h
        sign = image;
        return
    end
    
    % RETURN CROPPED SIGN AREA
    sign = cropImage(image, x1, y1, x2, y2);
end

% =========================================================================
% PHASE 2 : DIGIT EXTRACTION
% Takes in image to be processed
% Calls 2 main steps:
%   1. Preprocess image for digit extraction
%   2. If digit detected: Isolate digit
% Returns leading detected digit
function digit = ExtractDigit(image)
    [preprocessedImage, digitData, detected] = Preprocess(image);
    if detected == 0
        digit = preprocessedImage;
        return
    end
    digit = IsolateDigit(preprocessedImage, digitData);
end

% =========================================================================
% PREPROCESSING FOR DIGIT ISOLATION
% Takes in image to process
% Preprocessing includes: converting to inverted binary image and defining
% bounding box area of interest properties
% Returns preprocessed image, realted detected digit property data and
% detected (T/F) variable
function [preprocessedImg, digitData, detected] = Preprocess(image)
    % INITIALIZE VALUES
    lwrTh = 3000;
    uprTh = 14000;
    binaryThreshold = 0.3;

    % GREYSCALE CONVERSION
    greyscaleImage = rgb2gray(image);
    
    % INCREASE IMAGE CONTRAST
    contrastGreyscaleImage = histeq(greyscaleImage); 
    
    % BINARIZE IMAGE AND INVERT
    binaryImage = imbinarize(contrastGreyscaleImage, binaryThreshold);
    invBinaryImage = ~binaryImage;
      
    % CHARACTERISTICS OF CONNECTED COMONENTS
    properties = regionprops('table', invBinaryImage, 'Area', 'Centroid', 'BoundingBox'); 
    
    % PIXEL AREA THRESHOLDING
    lowerTh = properties.Area < lwrTh;
    upperTh = properties.Area > uprTh;
    properties(logical(lowerTh + upperTh),:) = [];
    
    % CHECK FOR IMAGE DETECTION
    if height(properties) == 0 || isempty(properties)
        % NO DIGIT DETECTED: RETURN IMAGE
        detected = 0;
    else
        % DIGIT DETECTED: RETURN DIGIT ISOLATION DATA
        detected = 1;
    end
    
    preprocessedImg = invBinaryImage;
    digitData = properties;
end

% =========================================================================
% DIGIT ISOLATION
% Takes in preprocessed image and realted detected digit property data
% Crops image to digit of interests bounding box region
% Returns isolated digit inverted binarized image
function digit = IsolateDigit(preprocessedImage, digitData)
    minDistance = 650; % Running closest distance between region centroid and reference centroid
    boundingBoxRegion = [];       % Bounding box of digit

    % LOOP OVER BOUNDING BOX REGIONS
    for i = 1:height(digitData)
        distance = norm([80 130]-digitData.Centroid(i));
        if distance < minDistance
            minDistance = distance;
            boundingBoxRegion = digitData.BoundingBox(i,:);
        end
    end

    % GET BOUNDING BOX X & Y COORDS FOR DIGITS
    x1 = ceil(boundingBoxRegion(1,1)); 
    x2 = x1 + floor(boundingBoxRegion(1,3)-1); 
    y1 = ceil(boundingBoxRegion(1,2));
    y2 = y1 + floor(boundingBoxRegion(1,4)-1);

    % RETURN ISOLATED DIGIT
    digit = cropImage(preprocessedImage, x1, y1, x2, y2);
end

% =========================================================================
% PHASE 3 : SPEED LIMIT CLASSIFICATION
% Takes in leading digit to be identified
% Calls 2 main steps:
%   1. Classify digit
%   2. Extract Speed Limit
% Returns predicted speed limit and confidence with which it is predicted
function [speedLimit, confidence] = ClassifyObject (digit)
    confidences = ClassifyDigit(digit);
    [speedLimit,confidence] = SpeedLimitExtraction(confidences);
end

% =========================================================================
% DIGIT CLASSIFICATION
% Takes in isolated digit image
% Compares with gold standard isolated inverted binary images
% Calulates confidence for each comparison
% Returns array of confidences for each digit comparison
function confidences = ClassifyDigit(digit)
    % INITIALIZE
    dims = [170 130];
    confidences = zeros(1,5);
    
    % GOLD STANDARD DATA
    location = 'Datasets/GoldStandardDigits/';
    goldStandFiles = dir(fullfile(location,'*.jpg'));
    imgCount = length(goldStandFiles);
    
    % RESIZE
    if isempty(digit)
        return
    end
    digit = imresize(digit, dims);
    
    % LOOP OVER GOLD STANDARD ISOLATED DIGIT IMAGES
    for i = 1:imgCount
        % DATA
        file = fullfile(location, goldStandFiles(i).name);
        goldStandDigit = imread(file);
        
        % COMPARE GOLD STANDARD IMAGE WITH BINARY INVERTED ISOLATED DIGIT
        output = imsubtract(logical(goldStandDigit), digit);
        
        % CALCULATE CONFIDENCE IN COMPARISON
        confidences(i) = 100*(sum(output(:)==0)/numel(output(:)));
    end
    
end

% =========================================================================
% SPEED LIMIT EXTRACTION
% Takes in confidences generated in classifcaion step and finds maximum
% confidence value and corresponding speed limit
% Returns speed limit and maximum confidence value
function [speedLimit, confidence] = SpeedLimitExtraction (confidences)
    % CLASSES
    speedLimits = {100 100 20 30 40 50 60 80};
    
    % HIGHEST CONFIDENCE MATCH
    [~, index] = max(confidences);
    confidence = max(confidences);
    
    % SPEED LIMIT EXTRACTION BASED ON HIGHEST CONFIDENCE
    speedLimit = speedLimits{index};
end
