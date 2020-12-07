%% �����������
[FileName,PathName,FilterIndex] = uigetfile( ...
    {'*.jpg';'*.bmp','*.png'},'�뵼�����ͼƬ','*.png','MultiSelect','on');
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
%% �Բ����������з���
preTestLabel = svmpredict(TestLabel, TestData, model);
assignin('base','TestLabel',TestLabel);
assignin('base','preTestLabel',preTestLabel);
TestLabel'
preTestLabel'

