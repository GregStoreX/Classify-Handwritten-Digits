clc,clear;
load mnist;
[row_num,col_num]=size(x);
masks = zeros(5,5,16);
% edge masks
masks(:,:,1)=reshape([-1,0,1,0,0,-1,0,1,0,0,-1,0,1,0,0,-1,0,1,0,0,-1,0,1,0,0],5,5);
masks(:,:,2)=reshape([0,0,1,0,-1,0,0,1,0,-1,0,0,1,0,-1,0,0,1,0,-1,0,0,1,0,-1],5,5);
masks(:,:,3)=masks(:,:,1)';
masks(:,:,4)=masks(:,:,2)';
masks(:,:,5)=reshape([0,0,-2,0,1,0,-1,0,1,0,-2,0,1,0,0,0,1,0,0,0,1,0,0,0,0],5,5);
masks(:,:,6)=reshape([0,0,0,0,1,0,0,0,1,0,0,0,1,0,-2,0,1,0,-1,0,1,0,-2,0,0],5,5);
masks(:,:,7)=reshape([1,0,-2,0,0,0,1,0,-1,0,0,0,1,0,-2,0,0,0,1,0,0,0,0,0,1],5,5);
masks(:,:,8)=masks(:,:,7)';

%  end stop masks
masks(:,:,9)=reshape([0,-1,-1,-1,-1,-8,0,0,0,0,-8,2,2,2,2,-8,0,0,0,0,0,-1,-1,-1,-1],5,5);
masks(:,:,10)=reshape([-1,-1,-1,-1,0,0,0,0,0,-8,2,2,2,2,-8,0,0,0,0,-8,-1,-1,-1,-1,0],5,5);
masks(:,:,11)=masks(:,:,9)';
masks(:,:,12)=masks(:,:,10)';
masks(:,:,13)=reshape([0,0,-2,0,2,0,-1,0,2,0,-1,0,2,0,-2,-8,2,0,-1,0,-8,-8,-1,0,0],5,5);
masks(:,:,14)=masks(:,:,13)';
masks(:,:,15)=reshape([-8,-8,-1,0,0,-8,2,0,-1,0,-1,0,2,0,-2,0,-1,0,2,0,0,0,-2,0,2],5,5);
masks(:,:,16)=reshape([2,0,-2,0,0,0,2,0,-1,0,-2,0,2,0,-1,0,-1,0,2,-8,0,0,-1,-8,-8],5,5);
tic;
features = zeros(row_num,64);
for i=1:row_num
    piece = reshape(x(i,:),28,28)';
    for j=1:16
        mask = masks(:,:,j);
        conv_result = conv2(piece,mask,'valid');
        pooled = zeros(2,2);
        pooled(1,1)=max(max(conv_result(1:12,1:12)));
        pooled(1,2)=max(max(conv_result(1:12,13:24)));
        pooled(2,1)=max(max(conv_result(13:24,1:12)));
        pooled(2,2)=max(max(conv_result(13:24,13:24)));
        features(i,(j-1)*4+1:(j-1)*4+4) = reshape(pooled,1,4);
    end
    if mod(i,1000)==0
       disp([num2str(i),' rows finished']);
    end
end
toc;