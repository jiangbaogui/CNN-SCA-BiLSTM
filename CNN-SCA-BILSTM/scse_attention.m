function output = scse_attention(input)
    % 输入：
    % input - 输入特征张量，大小为 [height, width, channels]

    % 计算通道注意力
    channel_attention = mean(input, [1, 2]); % 在空间维度上求平均值
    channel_attention = sigmoid(channel_attention); % 使用Sigmoid激活函数
    channel_attention = reshape(channel_attention, [1, 1, numel(channel_attention)]); % 将通道注意力广播到原始形状

    % 计算空间注意力
    spatial_attention = mean(input, 3); % 在通道维度上求平均值
    spatial_attention = sigmoid(spatial_attention); % 使用Sigmoid激活函数
    spatial_attention = reshape(spatial_attention, [size(input, 1), size(input, 2), 1]); % 将空间注意力广播到原始形状

    % 将通道和空间注意力相乘，得到最终的SCSE注意力
    attention = channel_attention .* spatial_attention;

    % 将输入特征与注意力相乘得到输出
    output = input .* attention;
end

function result = sigmoid(x)
    result = 1 ./ (1 + exp(-x));
end


