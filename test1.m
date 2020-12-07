clear;clc;

% train
train_fileName='C:\Users\lee\Desktop\code\�˹�������ģʽʶ��\artificial_intelligence\code\num_recog\Mnist-image\0and1Train';
train_Files = dir(strcat(train_fileName,'*.png'));
LengthFiles = length(train_Files);
train_img_arr=[];
train_label(1:500)=ones(1,500)*0;
train_label(501:1000)=ones(1,500)*1;
train_label=train_label'; %ѵ���ı�ǩ ת��
for i = 1:LengthFiles
    srcimg = imread(strcat(fileName,Files(i).name));
    im=rgb2gray(srcimg);
    bwimg=im2bw(im,5/255); %��ֵ��
    img_arr = reshape(bwimg, 1, prod(size(bwimg))); %ͼ��չ��Ϊһ��
    img_arr=double(img_arr);
    train_img_arr=[train_img_arr;img_arr]; %��������double����Ȼ�ᱨ��
end
model = svmtrain(train_label, train_img_arr, '-s 0 -c 0.5 -t 2 -g 1 -r 1 -d 3');
save('shouxie_model','model');
% [predict_label, accuracy, dec_values] =svmpredict(heart_scale_label, heart_scale_inst, model);

% test
test_filename='C:\Users\lee\Desktop\code\�˹�������ģʽʶ��\artificial_intelligence\code\num_recog\Mnist-image\test';
test_Files = dir(strcat(test_filename,'*.png'));
test_LengthFiles = length(test_Files);
test_label(1:100)=ones(1,100)'*0;
test_label(101:200)=ones(1,100)'*1;
test_label=double(test_label'); %���Եı�ǩ
test_img_arr=[];
for i = 1:test_LengthFiles
    test_img = imread(strcat(test_filename,test_Files(i).name));
    test_im=rgb2gray(test_img);
    test_bwimg=im2bw(test_im,5/255);
    img_arr1 = reshape(test_bwimg, 1, prod(size(test_bwimg)));
    img_arr1=double(img_arr1);
    test_img_arr=[test_img_arr;img_arr1];
end
[predict_label1, accuracy, dec_values] =svmpredict(test_label, test_img_arr, model);
