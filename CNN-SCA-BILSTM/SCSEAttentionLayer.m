classdef SCSEAttentionLayer < nnet.layer.Layer
    methods
        function layer = SCSEAttentionLayer(name)
            % 构造函数
            % name - 自定义层的名称
            layer.Name = name;
            layer.Description = 'SCSE Attention Layer';
        end

        function output = predict(~, input)
            % 预测方法
            % input - 输入数据
            channel_attention = mean(mean(input, 2), 1);
            channel_attention = sigmoid(channel_attention);
            channel_attention = reshape(channel_attention, [1, 1, numel(channel_attention)]);

            spatial_attention = mean(input, 3);
            spatial_attention = sigmoid(spatial_attention);
            spatial_attention = reshape(spatial_attention, size(input, 1), size(input, 2), 1);

            attention = channel_attention .* spatial_attention;
            output = input .* attention;
        end
    end
end

function result = sigmoid(x)
    result = 1 ./ (1 + exp(-x));
end
