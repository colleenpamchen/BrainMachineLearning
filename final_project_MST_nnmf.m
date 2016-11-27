%% NNMF of MNIST dataset approximation of MST neurons 
% I THINK IM GOING CRAZY! don't start a new project on GitHub, and then 
% git init the local directory. I did this last time and it was the wrong
% thing! 
% setting up MNIST dataset, images, and labels 
idx = 5;

filename_images='train-images-idx3-ubyte';
% function images = loadMNISTImages(filename)
% loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
% the raw MNIST images

fp = fopen(filename_images, 'rb');
assert(fp ~= -1, ['Could not open ', filename_images, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename_images, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

imshow(reshape(images(:,idx), 28, 28)); % each column is an digit, which has 784 (rows) components, represented imshow() as 28x28  
% end

% function labels = loadMNISTLabels(filename)
% loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
% the labels for the MNIST images

filename_labels='train-labels-idx1-ubyte';
fp = fopen(filename_labels, 'rb');
assert(fp ~= -1, ['Could not open ', filename_labels, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename_labels, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);
% end
disp(labels(idx))




% notes below here: 
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
 
% We are using display_network from the autoencoder code
% display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));


n=10;
k=1;
for i = 1:n
%     figure;
subplot(n,1,i) 
% imshow(reshape(images(:,i*k), 28, 28));
imshow(reshape(images(:,i), 28, 28));
end

idx=100;
imshow(reshape(images(:,idx), 28, 28));




function [] = saveMNISTImages(images, n, k)
% saveMNISImages Saves the first every k-th image of the MNIST training
% data set up to n images.

    for i = 1: n
        imwrite(reshape(images(:,i*k), 28, 28), strcat('MNIST/', num2str(i*k), '.png'));
    end;
end