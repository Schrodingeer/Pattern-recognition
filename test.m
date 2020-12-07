%% 载入测试样本
[FileName,PathName,FilterIndex] = uigetfile( ...
    {'*.jpg';'*.bmp','*.png'},'请导入测试图片','*.png','MultiSelect','on');
if ~FilterIndex
    return;
end
num_train = length(FileName);
TestData = zeros(num_train,16*16);
TestLabel = zeros(num_train,1);
for k = 1:num_train
    pic = imread([PathName,FileName{k}]);
    pic = pic_preprocess(pic);

    TestData(k,:) = double(pic(:)');
    TestLabel(k) = str2double(FileName{k}(4));
end
%% 对测试样本进行分类
preTestLabel = svmpredict(TestLabel, TestData, model);
assignin('base','TestLabel',TestLabel);
assignin('base','preTestLabel',preTestLabel);
TestLabel'
preTestLabel'

