close all
clear all
clc

img_path='G:\Fuzzy_Active_Contour\Solar_Image_2017\';

Files=dir([img_path,'*.*']);
for k=3:length(Files)
FileNames=Files(k).name;
end

img_1=imread([img_path,Files(32).name]);
img1=img_1;
imshow(img1)
% [img_m,mask1]=cosegmentation_ini(img1,0.55);

for imgk=32%:length(Files)

close all    
    
Files(imgk).name    
img_name=Files(imgk).name;

% [img_name,img_path] = ([img_path,'20170130_175629_1024_0193.jpg']);
name_1=([img_path,img_name]);
[filepath,name,ext] = fileparts(name_1); 


%%
tic
%read the images
img_1=imread(name_1);
img1=img_1;
[img_m,mask1]=cosegmentation_ini(img1,0.55);
% img_1=img_m;

%show the images
% % figure(1);
% % imshow(img_1);

P = rgb2gray((img_1));
%P=P^(1.2);
P = double(P);
%% 

num_iter=10;
layer=1;
mu=1;
width=1.2;%for level-set band determinition
alpha3=0.6%0.5%0.5%0.5; % for F1
alpha4=1%0.001%0.1%1; % for F2
delt=5; % for initial curve
nn=5;
m=20; % weighting exponent of each fuzzy membership
%% 

mask2 = zeros(size(P));
% mask(delt:end-delt,delt:end-delt) = 1;
mask2=maskcircle2(img1,'large');
mask2 = mask2(:,:,1);

phi0 = bwdist(mask1)-bwdist(1-mask1)+im2double(mask1)-.5; 
phi1 = bwdist(mask1)-bwdist(1-mask1)+im2double(mask1)-.5; 
%   initial force, set to eps to avoid division by zeros
force = eps; 
%-- End Initialization

%-- Display settings
figure();
subplot(2,2,1); imshow(img_1); title('Input Image');
subplot(2,2,2); contour(flipud(phi0), [0 0], 'r','LineWidth',1);hold on;
% contour(flipud(phi1), [0 0], 'r','LineWidth',1);title('initial contour');
subplot(2,2,3); title('Segmentation');
%-- End Display original image and mask

%-- Main loop
inidx = find(phi0>=0); % frontground index
outidx = find(phi0<0); % background index

[x y]=size(P);

for i=1:x
    for j=1:y
        if phi0(i,j)>=0
            u(i,j)=0.6;
        else
            u(i,j)=0.4;
        end
    end
end

L = im2double(P(:,:,1));

nc1=(L.*(u).^m);
dc1=((u).^m);    
nc1=sum(sum(nc1));
dc1=sum(sum(dc1));


nc2=(L.*((1-u)).^m);
dc2=((1-u).^m);
nc2=sum(sum(nc2));
dc2=sum(sum(dc2));

c1=nc1/dc1; % average inside of Phi0
c2=nc2/dc2; % average outside of Phi0

v=u;
size(v);
for i=1:layer
    L = im2double(P(:,:,i)); % get one image component
end
force_image=((v.^m).*(L-c1).^2)+(((1-v).^m).*(L-c2).^2);
F1=sum(sum(force_image));

%% 

%-- Main loop

for n=1:num_iter
    % intermediate output
    if(mod(n,1) == 0) 
        for j = 1:size(phi0,3)
            phi_{j} = phi0(:,:,j);
        end
        imshow(img_1,'initialmagnification','fit','displayrange',[0 255]);
        hold on;
        
        if size(phi0,3) == 1
            contour(phi_{1}, [0 0], 'r','LineWidth',4);
            contour(phi_{1}, [0 0], 'g','LineWidth',1.3);
        else
            contour(phi_{1}, [0 0], 'r','LineWidth',4);            
            contour(phi_{1}, [0 0], 'x','LineWidth',1.3);
            contour(phi_{2}, [0 0], 'g','LineWidth',4);
            contour(phi_{2}, [0 0], 'x','LineWidth',1.3);
        end
        hold off;
        title([num2str(n) ' Iterations']); 
        drawnow;
        %         showphi(P,phi0,n);  
    end;
    
%     if (its==1)||(its==500)||(its==1000)
%         figure(20);
%         imshow(img_1,'initialmagnification','fit','displayrange',[0 255]);
%         hold on;
%         contour(phi_{1}, [0 0], 'r','LineWidth',4);
%         contour(phi_{1}, [0 0], 'g','LineWidth',1.3);
%         saveas(gcf,[img_path,'\cont\',name_1,'con_s',num2str(n),'.bmp'],'bmp')
%         title(num2str(its));
%     end
    
    inidx = find(phi0>=0); % frontground index
    outidx = find(phi0<0); % background index
%     force_image = 0; % initial image force for each layer 
    for i=1:layer
        L = im2double(P(:,:,i)); % get one image component
        
        % sum Image Force on all components (used for vector image)
        % if 'chan' is applied, this loop become one sigle code as a
        % result of layer = 1
    end
    
    nc1=(L.*(u).^m);
    dc1=((u).^m);    
    nc1=sum(sum(nc1));
    dc1=sum(sum(dc1));
    
    
    nc2=(L.*((1-u)).^m);
    dc2=((1-u).^m);
    nc2=sum(sum(nc2));
    dc2=sum(sum(dc2));
    
    
    c1=nc1/dc1; % average inside of Phi0
    c2=nc2/dc2; % average outside of Phi0
    
           
    nr=(alpha3*(L-c1).^2);
    dr=(alpha4*(L-c2).^2)+eps;
    nu=1./(1+(nr./dr).^(1/(m-1)));
    
    s1=sum(sum(u.^(m)));
    s2=sum(sum((1-u).^(m)));
        
    su1=s1*ones(x,y);
    su2=s2*ones(x,y);
    dmu1=((nu).^(m))-((u).^(m));
    dmu2=((1-nu).^(m))-((1-u).^(m));
    sd1=su1+dmu1;
    sd2=su2+dmu2;
    
    deltf=(((alpha3*s1).*dmu1.*(L-c1).^2)./(sd1))+(((alpha4*s2).*dmu2.*(L-c2).^2)./(sd2));
        
    for i=1:x
        for j=1:y
            if deltf(i,j)<0
                u(i,j)=nu(i,j);
            else
                u(i,j)=u(i,j);
            end
        end
    end
    
    for i=1:x
        for j=1:y
            if u(i,j)>=0.5
                phi0(i,j)=1;
            else
                phi0(i,j)=-1;
            end
        end
    end
    
    
    v=nu;
    force_image=((v.^m).*(L-c1).^2)+(((1-v).^m).*(L-c2).^2);
    F=sum(sum(force_image));
   
    if F==F1
        break;
    end
    
    F1=F;
end   

phi1=phi1;
[x,y]=size(phi1);
for i=1:x
    for j=1:y
        if ((phi0(i,j)>=0)&&(phi1(i,j)<0))
            phi2(i,j)=-phi0(i,j);
        elseif ((phi0(i,j)>=0)&&(phi1(i,j)>=0))
            phi2(i,j)=1;
        elseif (phi0(i,j)<0 && phi1(i,j)<0)
            phi2(i,j)=1;
        end
    end
end

for j = 1:size(phi1,3)
    phi0_{j} = phi0(:,:,j);
    phi1_{j} = phi1(:,:,j);
    phi2_{j} = phi2(:,:,j);
end

imshow(img_1,'initialmagnification','fit','displayrange',[0 255]);
hold on;
if size(phi0,3) == 1
    contour(phi0_{1}, [0 0], 'r','LineWidth',4);
    contour(phi0_{1}, [0 0], 'g','LineWidth',1.3);
    contour(phi1_{1}, [0 0], 'r','LineWidth',4);
    contour(phi1_{1}, [0 0], 'g','LineWidth',1.3);
else
    contour(phi0_{1}, [0 0], 'r','LineWidth',4);
    contour(phi0_{1}, [0 0], 'x','LineWidth',1.3);
    contour(phi0_{2}, [0 0], 'g','LineWidth',4);
    contour(phi0_{2}, [0 0], 'x','LineWidth',1.3);
    contour(phi1_{1}, [0 0], 'r','LineWidth',4); 
    contour(phi1_{1}, [0 0], 'x','LineWidth',1.3);
    contour(phi1_{2}, [0 0], 'g','LineWidth',4);
    contour(phi1_{2}, [0 0], 'x','LineWidth',1.3);
end
hold off;
title([num2str(n) ' Iterations']); 
drawnow;

%make mask from SDF
seg = phi2<0; %-- Get mask from levelset
subplot(2,2,4); imshow(seg); title('Global Region-Based Segmentation');
result_img_path='G:\Fuzzy_Active_Contour\Result\Output\';
% imwrite(seg,[img_path,'bmp\',name,'.bmp'],'bmp');
% imwrite(seg,[img_path,'jpg\',name,'.jpg'],'jpg');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%the first image
mask1_1=zeros(size(phi2,1),size(phi2,2));
mask1_1(find(phi2<0))=1;
mask1_1=logical(mask1_1);
imshow(mask1_1)
% imwrite(mask1_1,['G:\Fuzzy_Active_Contour\Result\Output\bmp\',name,'mask1_ori.bmp'],'bmp');

img1_temp=[];
img1_temp=img1;
for j=1:size(img1,1)
    for k=1:size(img1,2)
        if mask1_1(j,k)==0
            for i=1:size(img1,3)
                img1_temp(j,k,i)=255;
            end
        end
    end
end
% imwrite(img1_temp,['G:\Fuzzy_Active_Contour\Result\Output\bmp\',name,'seg1_ori.bmp'],'bmp');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%post-processing
L_1= bwlabel(mask1_1);
max_region_1=max(max(L_1));
if max_region_1~=1
    si=[];
    for i=1:max_region_1
        si(i)=size(find(L_1==i),1);
    end
    [max_s,index]=sort(si,'descend');
    T=max_s(1)*0.05;
    for i=1:max_region_1
        if si(i)<=T
            mask1_1(find(L_1==i))=0;
        end
    end
end
imwrite(mask1_1,['G:\Fuzzy_Active_Contour\Result\Mask\bmp\',name,'.bmp'],'bmp');
imwrite(mask1_1,['G:\Fuzzy_Active_Contour\Result\Mask\jpg\',name,'.jpg'],'jpg');
phi11=mask1_1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%save the results
img1_temp=[];
img1_temp=img1;
for j=1:size(img1,1)
    for k=1:size(img1,2)
        if mask1_1(j,k)==0
            for i=1:size(img1,3)
                img1_temp(j,k,i)=255;
            end
        end
    end
end
imwrite(img1_temp,['G:\Fuzzy_Active_Contour\Result\Output\bmp\',name,'.bmp'],'bmp');
imwrite(img1_temp,['G:\Fuzzy_Active_Contour\Result\Output\jpg\',name,'.jpg'],'jpg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

toc

figure
imshow(img_1,'initialmagnification','fit','displayrange',[0 255]);
truesize
hold on;
contour(phi0_{1}, [0 0], 'r','LineWidth',4);
contour(phi0_{1}, [0 0], 'g','LineWidth',1.3);
contour(phi1_{1}, [0 0], 'r','LineWidth',4);
contour(phi1_{1}, [0 0], 'g','LineWidth',1.3);
saveas(gcf,['G:\Fuzzy_Active_Contour\Result\Contour\bmp\',name,'.bmp'],'bmp')
saveas(gcf,['G:\Fuzzy_Active_Contour\Result\Contour\jpg\',name,'.jpg'],'jpg')


figure
phi3 = bwdist(mask1_1)-bwdist(1-mask1_1)+im2double(mask1_1)-.5;
for j = 1:size(phi3,3)
    phi3_{j} = phi3(:,:,j);
end
imshow(img_1,'initialmagnification','fit','displayrange',[0 255]);
truesize
hold on;
contour(phi3_{1}, [0 0], 'r','LineWidth',4);
contour(phi3_{1}, [0 0], 'g','LineWidth',1.3);
saveas(gcf,['G:\Fuzzy_Active_Contour\Result\Final_Contour\bmp\',name,'.bmp'],'bmp')
saveas(gcf,['G:\Fuzzy_Active_Contour\Result\Final_Contour\jpg\',name,'.jpg'],'jpg')

end
