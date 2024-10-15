clc;
clear;
warning off;

%% 导入数据
Data = table2array(readtable("***.xlsx"));

%% 划分训练集和测试集
TotalSamples = size(Data, 1); % 样本总数
InPut_num = 1:10;  % 输入特征列
OutPut_num = 11;   % 输出响应列
Temp = randperm(TotalSamples);

% 按7:3比例划分
Train_Size = round(0.7 * TotalSamples);  % 训练集样本数量

% 分配训练集和测试集
Train_InPut = Data(Temp(1:Train_Size), InPut_num);  % 训练输入
Train_OutPut = Data(Temp(1:Train_Size), OutPut_num);  % 训练输出
Test_InPut = Data(Temp(Train_Size+1:end), InPut_num);  % 测试输入
Test_OutPut = Data(Temp(Train_Size+1:end), OutPut_num);  % 测试输出

% 清理临时变量
clear Temp TotalSamples Train_Size;
%% 数据归一化
[~, Ps.Input] = mapminmax([Train_InPut;Test_InPut]',-1,1); 
Train_InPut = mapminmax('apply',Train_InPut',Ps.Input);
Test_InPut = mapminmax('apply',Test_InPut',Ps.Input);
[~, Ps.Output] = mapminmax([Train_OutPut;Test_OutPut]',-1,1);
Train_OutPut = mapminmax('apply',Train_OutPut',Ps.Output);
Test_OutPut = mapminmax('apply',Test_OutPut',Ps.Output);
Temp_TrI = cell(size(Train_InPut,2),1);
Temp_TrO = cell(size(Train_OutPut,2),1);
Temp_TeI = cell(size(Test_InPut,2),1);
Temp_TeO = cell(size(Test_OutPut,2),1);

% 转为cell格式
for i = 1:size(Train_InPut,2)
    Temp_TrI{i} = Train_InPut(:,i);
    Temp_TrO{i} = Train_OutPut(:,i);
end
Train_InPut = Temp_TrI;
Train_OutPut = Temp_TrO;

for i = 1:size(Test_InPut,2)
    Temp_TeI{i} = Test_InPut(:,i);
    Temp_TeO{i} = Test_OutPut(:,i);
end
Test_InPut = Temp_TeI;
Test_OutPut = Temp_TeO;

clear Temp_TrI Temp_TrO Temp_TeI Temp_TeO;

%% 设置网络输入参数
numFeatures = length(InPut_num); % 输入特征个数
numResponses = length(OutPut_num); % 输出特征个数
numHiddenUnits = 0; % 隐含层神经元个数
Train_number =0; % 训练次数
dorp_rate = 0; % 遗忘率(可以防止过拟合，0.2表示遗忘20%)
LearnRate = 0; % 学习率
filterSize = 0; % 卷积核大小
numFilters = 0; % 卷积核个数
poolSize = 0; % 池化层大小
%% 构建CNN-biLSTM网络
layer = [ ...
    sequenceInputLayer(numFeatures) 
    convolution1dLayer(filterSize,numFilters,'Padding','same') % 卷积神经网络1
    reluLayer 
    layerNormalizationLayer 
    convolution1dLayer(filterSize/2,numFilters/2,'Padding','same') % 卷积神经网络2
    reluLayer 
    layerNormalizationLayer 
    maxPooling1dLayer(poolSize,'Padding','same'); 
    flattenLayer
    SCSEAttentionLayer('scse_attention')%SCA
    bilstmLayer(numHiddenUnits) 
    fullyConnectedLayer(numResponses) 
    regressionLayer]; 

lgraph = layerGraph(layer);
figure(10);
plot(lgraph);
title("网络结构展示");
%% 网络训练设置
options = trainingOptions('adam', ... % 求解器设置
    'MaxEpochs',Train_number, ... % 最大迭代次数
    'GradientThreshold',1, ... % 防止梯度爆炸
    'InitialLearnRate',LearnRate, ... % 学习率
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',Train_number/2, ... % 学习率下降轮次
    'LearnRateDropFactor',0.5, ... % 学习率下降倍数
    'Verbose',0, ...
    'Plots','training-progress');

%% 开始训练网络
net = trainNetwork(Train_InPut,Train_OutPut,layer,options);

%% 网络测试
TPred = predict(net,Train_InPut); % 输出训练集预测值
YPred = predict(net,Test_InPut); % 输出测试集预测值

%% 反归一化
True_Train = []; % 训练集真实值
Predict_Train = []; % 训练集预测值
True_Test = []; % 测试集真实值
Predicte_Test = []; % 测试集预测值

% 反归一化训练集
for i = 1:size(Train_InPut,1)
    True_Train = [True_Train,mapminmax('reverse',Train_OutPut{i},Ps.Output)];
    Predict_Train = [Predict_Train,mapminmax('reverse',TPred{i},Ps.Output)];
end
Predict_Train = double(Predict_Train);

% 反归一化测试集
for i = 1:size(Test_OutPut,1)
    True_Test = [True_Test,mapminmax('reverse',Test_OutPut{i},Ps.Output)];
    Predicte_Test = [Predicte_Test,mapminmax('reverse',YPred{i},Ps.Output)];
end
Predicte_Test = double(Predicte_Test);

%% 误差值评价值输出
% RMSE 
RMSE1 = sqrt(mean((True_Train-Predict_Train).^2));
RMSE2 = sqrt(mean((True_Test-Predicte_Test).^2));
disp(['训练集数据的RMSE为：', num2str(RMSE1)]);
disp(['测试集数据的RMSE为：', num2str(RMSE2)]);

% R2指标
R1 = 1 - norm(True_Train - Predict_Train)^2 / norm(Predict_Train - mean(True_Train))^2;
R2 = 1 - norm(True_Test  - Predicte_Test)^2 / norm(Predicte_Test  - mean(True_Test ))^2;
disp(['训练集数据的R2误差为：', num2str(R1)]);
disp(['测试集数据的R2误差为：', num2str(R2)]);

% MAE
mae1 = sum(abs(Predict_Train - True_Train), 2)' ./ length(True_Train);
mae2 = sum(abs(Predicte_Test - True_Test ), 2)' ./ length(True_Test);
disp(['训练集数据的MAE为：', num2str(mae1)]);
disp(['测试集数据的MAE为：', num2str(mae2)]);


