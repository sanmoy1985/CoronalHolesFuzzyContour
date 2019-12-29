function [img2_temp,m]=contour_initialization(I,th)

% close all
% clear all

figure,imshow(I)
A=(I);

%https://stackoverflow.com/questions/27171527/circle-detection-from-gray-level-image-in-matlab
%// Some pre-processing. Treshold image and dilate it.
B = im2bw(A,th);

se = strel('disk',2);

C = imdilate(B,se);

D = bwareaopen(C,1000);

%// Here imfill is not necessary but you might find it useful in other situations.
E = imfill(D,'holes');

%// Detect edges
F = edge(E);

%// circle_hough from the File Exchange.

%// This code is based on Andrey's answer here:
%https://dsp.stackexchange.com/questions/5930/find-circle-in-noisy-data.

%// Generate range of radii.
%  radii = 200:10:650;
radii = 50:10:450;%Last
% radii =410;

h = circle_hough(F, radii,'same');
[~,maxIndex] = max(h(:));
[i,j,k] = ind2sub(size(h), maxIndex);
radius = radii(k);
center.x = j;
center.y = i;

%// Generate circle to overlay
N = 200;

theta=linspace(0,2*pi,N);
rho=ones(1,N)*radius;

%Cartesian coordinates
[X,Y] = pol2cart(theta,rho); 

figure;

subplot(2,2,1)
imshow(B);
title('Thresholded image  (B)','FontSize',16)

subplot(2,2,2)
imshow(E);
title('Filled image (E)','FontSize',16)

subplot(2,2,3)
imshow(F);hold on

plot(center.x-X,center.y-Y,'r-','linewidth',2);

title('Edge image + circle (F)','FontSize',16)

subplot(2,2,4)
imshow(A);hold on
plot(center.x-X,center.y-Y,'r-','linewidth',2);
title('Original image + circle (A)','FontSize',16)

temp = double(I(:,:,1));

m = zeros(size(temp));
[x,y] = meshgrid(1:min(size(temp,1),size(temp,2)));
n = zeros(size(x));
n((x-center.x).^2+(y-center.y).^2<radius.^2) = 1;
m(1:size(n,1),1:size(n,2)) = n;
figure,imshow(m)

img2_temp=[];
img2_temp=I;
for j=1:size(I,1)
    for k=1:size(I,2)
        if m(j,k)==0
            for i=1:size(I,3)
                img2_temp(j,k,i)=255;
            end
        end
    end
end
figure,imshow(img2_temp)