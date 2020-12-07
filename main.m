[FileName,PathName,FilterIndex] = uigetfile(...
    {'*.jpg';'*.bmp';'*.png'},'�뵼�����ͼƬ','*.png','MultiSelect','on');
if ~FilterIndex
    return;
end
num_train = length(FileName);
TrainData = zeros(num_train, 16*16);
TrainLabel = zeros(num_train, 1);
for k=1:num_train
    pic = imread([PathName, FileName{k}]);
    pic = pic_preprocess(pic);
    
    TrainData(k,:)=double(pic(:)');
    TrainLabel(k) = str2double(FileName{k}(1));
end

% ����֧��������

% ����GA��ز���
ga_option.mangen = 100;
ga_option.sizepop = 20;
ga_option.cbound = [0,100];
ga_option.gbound = [0, 100];
ga_option.v = 10;
ga_option.ggap = 0.9;
ga_option.maxgen = 100; % Ĭ��ֵ
ga_option.pCrossover = 0.4 % Ĭ��ֵ
ga_option.pMutation = 0.01 % Ĭ��ֵ
[bestCVaccuracy, bestc, bestg] = ...
    gaSVMcgForClass(TrainLabel, TrainData, ga_option)

% ѵ��
cmd = [ '-c', num2str(bestc),'-g',num2str(bestg)]; % t RBF
model = svmtrain(TrainLabel, TrainData);
% ��ѵ�����ϲ鿴ʶ������
preTrainLabel = svmpredict(TrainLabel, TrainData, model)




