classdef Recognizer_exp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                matlab.ui.Figure
        Menu                    matlab.ui.container.Menu
        PaintMenu               matlab.ui.container.Menu
        ContrastMenu            matlab.ui.container.Menu
        CNNMenu                 matlab.ui.container.Menu
        RandomForestMenu        matlab.ui.container.Menu
        AuthorRuixiMenu         matlab.ui.container.Menu
        GridLayout              matlab.ui.container.GridLayout
        RecognizeTestSetButton  matlab.ui.control.Button
        ChooseSet               matlab.ui.control.DropDown
        Label_2                 matlab.ui.control.Label
        ButtonGroup             matlab.ui.container.ButtonGroup
        Binarization            matlab.ui.control.RadioButton
        ContrastEnhancement     matlab.ui.control.RadioButton
        OriginalImage           matlab.ui.control.RadioButton
        LoadButton              matlab.ui.control.Button
        ResetButton             matlab.ui.control.Button
        TrainButton             matlab.ui.control.Button
        ChooseModel             matlab.ui.control.DropDown
        Label                   matlab.ui.control.Label
        TextArea                matlab.ui.control.TextArea
        RecognizeButton         matlab.ui.control.Button
        ClearButton             matlab.ui.control.Button
        PreviewAxes             matlab.ui.control.UIAxes
        UIAxes                  matlab.ui.control.UIAxes
    end

    % Copyright (C) 2025  Ruixi
    %
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU Lesser General Public License as published by
    % the Free Software Foundation, either version 3 of the License, or
    % (at your option) any later version.
    %
    % This program is distributed in the hope that it will be useful,
    % but WITHOUT ANY WARRANTY; without even the implied warranty of
    % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    % GNU Lesser General Public License for more details.
    %
    % You should have received a copy of the GNU Lesser General Public License
    % along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
    properties (Access = private)
        % 绘图
        imageData = ones(512, 512);
        isDrawing = false;
        previousPoint = [];
        imageHandle;

        brushRadius = 16;
        brushSettingsFigure;
        brushValueLabel;

        % CNN
        CNNModel = [];
        modelAccuracy = 0;
        trainingEpochs = 0;
        initialLearnRate = 0.001;

        cnnSettingsFigure;
        maxEpochsEdit;
        initialLearnRateEdit;
        
        % RF
        rfModel = [];
        rfAccuracy = 0;
        rfNumTrees = 100;

        rfSettingsFigure;
        numTreesEdit;

        % 数据与图像处理
        dataFolder = 'DataSets';
        trainFile = '';
        testFile = '';
        mappingFile = '';

        contrastStrength = 0.8;
        contrastSettingsFigure;
        contrastSlider;
        contrastValueLabel;

        % 应用状态
        isLoadingModel = false;
        lastLineIsResult = false;

        % CopyRight
        AuthorFigure;
    end

    methods (Access = private)
        % 更新预览图像
        function updatePreview(app)
            smallImage = app.preprocessImage();
            smallImage255 = uint8(smallImage * 255);
            % 根据选择的预处理方式处理图像
            selectedButton = app.ButtonGroup.SelectedObject;
            switch selectedButton.Text
                case '原图像'
                    processedImage = smallImage255;
                case '二值化'
                    binaryImage = imbinarize(smallImage255/255);
                    processedImage = uint8(binaryImage * 255);
                case '对比度增强'
                    imgDouble = im2double(smallImage255);
                    enhancedImg = (imgDouble - 0.5) * (1 + app.contrastStrength) + 0.5;
                    
                    %strength = app.contrastStrength;
                    %if strength < 0
                    %    factor = 1 + strength;
                    %    enhancedImg = imgDouble * factor;
                    %else
                    %    factor = 1 + strength;
                    %    enhancedImg = (imgDouble - 0.5) * factor + 0.5;
                    %end

                    enhancedImg = max(0, min(1, enhancedImg));
                    processedImage = im2uint8(enhancedImg);
                otherwise
                    processedImage = smallImage255;
            end

            % 在预览坐标轴显示处理后的图像
            cla(app.PreviewAxes);
            imshow(processedImage, 'Parent', app.PreviewAxes);
        end

        % 预处理图像：裁剪、缩放、反转颜色
        function processedImg = preprocessImage(app)
            [rows, cols] = find(app.imageData == 0);
            % 找到绘制内容的坐标（黑色笔迹为0）
            if isempty(rows)
                processedImg = zeros(28, 28); % 空图像
            else
                % 计算内容边界
                minRow = min(rows);
                maxRow = max(rows);
                minCol = min(cols);
                maxCol = max(cols);
        
                % 计算包围框尺寸并扩展为稍大的正方形区域（1.2倍）
                baseLength = max(maxRow - minRow + 1, maxCol - minCol + 1);
                expandedLength = round(baseLength * 1.2);
        
                % 计算内容中心
                centerRow = (minRow + maxRow) / 2;
                centerCol = (minCol + maxCol) / 2;
        
                % 理想裁剪区域（可能超出图像边界）
                idealStartRow = round(centerRow - expandedLength / 2);
                idealStartCol = round(centerCol - expandedLength / 2);
                idealEndRow = idealStartRow + expandedLength - 1;
                idealEndCol = idealStartCol + expandedLength - 1;
        
                % 获取实际在图像内的区域
                actualStartRow = max(1, idealStartRow);
                actualStartCol = max(1, idealStartCol);
                actualEndRow = min(512, idealEndRow);
                actualEndCol = min(512, idealEndCol);
        
                % 提取有效内容
                validRegion = app.imageData(actualStartRow:actualEndRow, actualStartCol:actualEndCol);
        
                % 创建 expandedLength × expandedLength 的白底图像
                paddedImage = ones(expandedLength, expandedLength);
        
                % 处理越界
                pasteRowStart = actualStartRow - idealStartRow + 1;
                pasteColStart = actualStartCol - idealStartCol + 1;
                pasteRowEnd = pasteRowStart + size(validRegion, 1) - 1;
                pasteColEnd = pasteColStart + size(validRegion, 2) - 1;
        
                % 将有效区域粘贴到居中位置
                paddedImage(pasteRowStart:pasteRowEnd, pasteColStart:pasteColEnd) = validRegion;
        
                % 缩小为28x28并反转颜色（0→1, 1→0，使笔迹为1，背景为0）
                processedImg = imresize(paddedImage, [28, 28]);
                processedImg = 1 - processedImg;
            end
        end
        
        % 训练CNN模型
        function trainCNNModel(app, XTrain, YTrain, XTest, YTest)
            app.TextArea.Value = {'开始训练CNN模型...'};
            drawnow;
        
            layers = [
                imageInputLayer([28 28 1], 'Name', 'input', 'Normalization', 'none')
                convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_1')
                batchNormalizationLayer('Name', 'bn_1')
                reluLayer('Name', 'relu_1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
                convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_2')
                batchNormalizationLayer('Name', 'bn_2')
                reluLayer('Name', 'relu_2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
                fullyConnectedLayer(128, 'Name', 'fc_1')
                reluLayer('Name', 'relu_3')
                dropoutLayer(0.5, 'Name', 'dropout')
                fullyConnectedLayer(numel(categories(YTrain)), 'Name', 'fc_output')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'classoutput')];
        
            options = trainingOptions('sgdm', ...
                'ExecutionEnvironment', 'auto', ...
                'MaxEpochs', app.trainingEpochs, ...
                'MiniBatchSize', 256, ...
                'InitialLearnRate', app.initialLearnRate, ...
                'Shuffle', 'every-epoch', ...
                'Plots', 'training-progress', ...
                'Verbose', false);
        
            app.CNNModel = trainNetwork(XTrain, YTrain, layers, options);
        
            YPred = classify(app.CNNModel, XTest);
            app.modelAccuracy = mean(YPred == YTest);
            
            % 保存模型
            datasetType = app.getDatasetType();
            modelsDir = fullfile(fileparts(mfilename('fullpath')), 'Models');
            if ~exist(modelsDir, 'dir')
                mkdir(modelsDir);
            end
            fileName = fullfile(modelsDir, sprintf('CNN_%s_Model.mat', datasetType));

            net = app.CNNModel;
            accuracy = app.modelAccuracy;
            epochs = app.trainingEpochs;
            save(fileName, 'net', 'accuracy', 'epochs', 'datasetType');
        
            % 显示训练结果
            outputStr = sprintf('模型: CNN\n训练次数: %d\n准确率: %.2f%%', app.trainingEpochs, app.modelAccuracy * 100);
            app.TextArea.Value = {outputStr};
            app.lastLineIsResult = false;
            
            % 启用识别按钮
            app.RecognizeButton.Enable = 'on';
            app.RecognizeTestSetButton.Enable = 'on';
        end

        % 训练随机森林模型
        function trainRFModel(app, XTrain, YTrain, XTest, YTest)
            if app.rfNumTrees <= 0
                app.rfNumTrees = 100; % 设置默认值
            end
            app.TextArea.Value = {'开始训练随机森林模型...'};
            drawnow;
        
            XTrainVec = reshape(XTrain, [], size(XTrain,4))';
            XTestVec = reshape(XTest, [], size(XTest,4))';
        
            app.rfModel = TreeBagger(app.rfNumTrees, XTrainVec, YTrain);
        
            YPred = predict(app.rfModel, XTestVec);
            app.rfAccuracy = mean(YPred == YTest);
        
            datasetType = app.getDatasetType();
            modelsDir = fullfile(fileparts(mfilename('fullpath')), 'Models');
            if ~exist(modelsDir, 'dir')
                mkdir(modelsDir);
            end
            fileName = fullfile(modelsDir, sprintf('RF_%s_Model.mat', datasetType));
        
            model = app.rfModel;
            accuracy = app.rfAccuracy;
            numTrees = app.rfNumTrees;
            % 保存模型和相关数据，由于随机森林模型较大。必须使用v7.3及以上的格式保存
            save(fileName, 'model', 'accuracy', 'numTrees', 'datasetType', '-v7.3');
            
            % 显示训练结果
            outputStr = sprintf('模型: Random Forest\n树数量: %d\n准确率: %.2f%%', app.rfNumTrees, app.rfAccuracy * 100);
            app.TextArea.Value = {outputStr};
            app.lastLineIsResult = false;
            
            % 启用识别按钮
            app.RecognizeButton.Enable = 'on';
            app.RecognizeTestSetButton.Enable = 'on';
        end
        
        % 加载预训练模型
        function loadModel(app, filePath)
            try
                app.TextArea.Value = {['正在加载: ' filePath]};
                drawnow;
        
                loadedData = load(filePath);
        
                app.isLoadingModel = true;
        
                % 保存当前模型状态
                savedCNNModel = app.CNNModel;
                savedRfModel = app.rfModel;

                % 根据数据集类型更新文件路径
                if isfield(loadedData, 'datasetType')
                    datasetType = loadedData.datasetType;
                    switch datasetType
                        case 'Digits'
                            app.ChooseSet.Value = '数字';
                        case 'Letters'
                            app.ChooseSet.Value = '字母';
                        case 'Balanced'
                            app.ChooseSet.Value = '数字和字母';
                    end
                    switch app.ChooseSet.Value
                        case '数字'
                            title(app.UIAxes, '请在白色方框内绘制数字');
                        case '字母'
                            title(app.UIAxes, '请在白色方框内绘制字母');
                        case '数字或字母'
                            title(app.UIAxes, '请在白色方框内绘制数字或字母');
                    end
                    % 更新数据文件路径
                    switch app.ChooseSet.Value
                        case '数字'
                            app.dataFolder = 'DataSets';
                            app.trainFile = fullfile(app.dataFolder, 'emnist-digits-train.csv');
                            app.testFile = fullfile(app.dataFolder, 'emnist-digits-test.csv');
                            app.mappingFile = fullfile(app.dataFolder, 'emnist-digits-mapping.txt');
                        case '字母'
                            app.dataFolder = 'DataSets';
                            app.trainFile = fullfile(app.dataFolder, 'emnist-letters-train.csv');
                            app.testFile = fullfile(app.dataFolder, 'emnist-letters-test.csv');
                            app.mappingFile = fullfile(app.dataFolder, 'emnist-letters-mapping.txt');
                        case '数字和字母'
                            app.dataFolder = 'DataSets';
                            app.trainFile = fullfile(app.dataFolder, 'emnist-balanced-train.csv');
                            app.testFile = fullfile(app.dataFolder, 'emnist-balanced-test.csv');
                            app.mappingFile = fullfile(app.dataFolder, 'emnist-balanced-mapping.txt');
                    end
                end
        
                % 恢复模型状态
                app.CNNModel = savedCNNModel;
                app.rfModel = savedRfModel;
        
                % 判断模型类型并加载
                if isfield(loadedData, 'model') % 随机森林模型
                    app.rfModel = loadedData.model;
                    app.CNNModel = [];
        
                    if isfield(loadedData, 'accuracy')
                        app.rfAccuracy = loadedData.accuracy;
                    else
                        app.rfAccuracy = 0;
                    end
        
                    if isfield(loadedData, 'numTrees')
                        app.rfNumTrees = loadedData.numTrees;
                    else
                        app.rfNumTrees = 100;
                    end
        
                    app.ChooseModel.Value = 'Random Forest';
        
                    modelInfo = sprintf('模型: Random Forest\n树数量: %d\n准确率: %.2f%%', ...
                        app.rfNumTrees, app.rfAccuracy * 100);
                    app.TextArea.Value = strsplit(modelInfo, '\n');
                    app.lastLineIsResult = false;
        
                    app.RecognizeButton.Enable = 'on';
                    app.RecognizeTestSetButton.Enable = 'on';
                elseif isfield(loadedData, 'net')   % CNN模型
                    app.CNNModel = loadedData.net;
                    app.rfModel = [];
        
                    if isfield(loadedData, 'accuracy')
                        app.modelAccuracy = loadedData.accuracy;
                    else
                        app.modelAccuracy = 0;
                    end
        
                    if isfield(loadedData, 'epochs')
                        app.trainingEpochs = loadedData.epochs;
                    else
                        app.trainingEpochs = 0;
                    end
        
                    app.ChooseModel.Value = 'CNN';
        
                    modelInfo = sprintf('模型: CNN\n训练次数: %d\n准确率: %.2f%%', ...
                        app.trainingEpochs, app.modelAccuracy * 100);
                    app.TextArea.Value = strsplit(modelInfo, '\n');
                    app.lastLineIsResult = false;
        
                    app.RecognizeButton.Enable = 'on';
                    app.RecognizeTestSetButton.Enable = 'on';
                else
                    app.TextArea.Value = {'错误: 文件中未找到有效模型'};
                    app.lastLineIsResult = true;
        
                    app.RecognizeButton.Enable = 'off';
                    app.RecognizeTestSetButton.Enable = 'off';
                end
            catch ME
                app.TextArea.Value = {['加载模型时出错: ' ME.message]};
                app.lastLineIsResult = true;
            end
        
            app.isLoadingModel = false;
        end
        
        % 识别手写字符
        function recognizeCharacter(app)
            try
                img_to_recognize = app.preprocessImage();
                
                % 使用CNN模型识别
                if ~isempty(app.CNNModel)
                    img_input = single(reshape(img_to_recognize, [28, 28, 1, 1]));
                    predictedLabel = classify(app.CNNModel, img_input);
                    resultStr = sprintf('识别结果: %s', char(predictedLabel));
                
                % 使用随机森林模型识别
                elseif ~isempty(app.rfModel)
                    img_vec = double(img_to_recognize(:)');
                    predictedLabel = predict(app.rfModel, img_vec);
                    resultStr = sprintf('识别结果: %s', char(predictedLabel));
                else
                    app.TextArea.Value = {'请先训练或加载一个模型！'};
                    app.lastLineIsResult = true;
                    return;
                end

                % 更新文本区域显示结果
                currentContent = app.TextArea.Value;
                if isempty(currentContent) || (ischar(currentContent) && isempty(currentContent))
                    app.TextArea.Value = {resultStr};
                else
                    if ischar(currentContent)
                        currentContent = {currentContent};
                    end
                    if app.lastLineIsResult
                        currentContent{end} = resultStr;
                    else
                        currentContent{end+1} = resultStr;
                    end
                    app.TextArea.Value = currentContent;
                end
                app.lastLineIsResult = true;
            catch ME
                currentContent = app.TextArea.Value;
                if ischar(currentContent)
                    currentContent = {currentContent};
                end
                app.TextArea.Value = [currentContent; {['识别时发生错误: ' ME.message]}];
                app.lastLineIsResult = true;
            end
        end
        % 创建测试结果显示界面
        function createTestResultsFigure(app, testImages, trueLabels, predictedLabels, accuracy)
            % 创建新窗口显示测试结果
            resultsFigure = uifigure('Name', '测试集识别结果');
            resultsFigure.Position = [200 200 800 600];

            grid = uigridlayout(resultsFigure);
            grid.ColumnWidth = {'1x', '1x', '1x', '1x'};
            grid.RowHeight = {'1x', '1x', '0.2x', '0.2x'};

            % 显示8个测试样本及其识别结果
            for i = 1:8
                row = ceil(i/4);
                col = mod(i-1, 4) + 1;

                ax = uiaxes(grid);
                ax.Layout.Row = row;
                ax.Layout.Column = col;

                imshow(squeeze(testImages(:, :, 1, i)), 'Parent', ax);
                title(ax, sprintf('真实: %s\n预测: %s', trueLabels{i}, predictedLabels{i}));

                if strcmp(trueLabels{i}, predictedLabels{i})
                    ax.Title.Color = [0 0.5 0];
                else
                    ax.Title.Color = [0.8 0 0];
                end
            end

            % 总体准确率
            accuracyLabel = uilabel(grid);
            accuracyLabel.Layout.Row = 3;
            accuracyLabel.Layout.Column = [1 4];
            accuracyLabel.Text = sprintf('总体准确率: %.1f%% (%d/8)', accuracy, round(accuracy*8/100));
            accuracyLabel.FontSize = 14;
            accuracyLabel.FontWeight = 'bold';
            accuracyLabel.HorizontalAlignment = 'center';
            
            % 关闭按钮
            closeButton = uibutton(grid, 'push');
            closeButton.Layout.Row = 4;
            closeButton.Layout.Column = [2 3];
            closeButton.Text = '关闭';
            closeButton.ButtonPushedFcn = @(src, event) delete(resultsFigure);
        end

        % 更新笔刷半径
        function updateBrushRadius(app, slider)

            % 获取滑块当前值并更新笔刷半径
            app.brushRadius = round(slider.Value);

            % 更新显示标签
            if ~isempty(app.brushValueLabel) && isvalid(app.brushValueLabel)
                app.brushValueLabel.Text = sprintf('当前半径: %d', app.brushRadius);
            end
        end

        % 关闭笔刷设置窗口
        function closeBrushSettings(app)
            delete(app.brushSettingsFigure);
            app.brushSettingsFigure = [];
        end

        % 获取当前选择的数据集类型（用于文件命名）
        function datasetType = getDatasetType(app)
            switch app.ChooseSet.Value
                case '数字'
                    datasetType = 'Digits';
                case '字母'
                    datasetType = 'Letters';
                case '数字和字母'
                    datasetType = 'Balanced';
                otherwise
                    datasetType = 'Unknown';
            end
        end
        % 应用CNN训练参数设置
        function applyCNNSettings(app)
            app.trainingEpochs = round(app.maxEpochsEdit.Value);
            app.initialLearnRate = app.initialLearnRateEdit.Value;
            delete(app.cnnSettingsFigure);
            app.cnnSettingsFigure = [];
            uialert(app.UIFigure, 'CNN参数已更新', '参数设置');
        end
        
        % 应用随机森林参数设置
        function applyRFSettings(app)
            newNumTrees = round(app.numTreesEdit.Value);
            if newNumTrees < 10
                newNumTrees = 10;
            elseif newNumTrees > 500
                newNumTrees = 500;
            end
            app.rfNumTrees = newNumTrees;
            delete(app.rfSettingsFigure);
            app.rfSettingsFigure = [];
            uialert(app.UIFigure, '随机森林参数已更新', '参数设置');
        end
        
        % 更新对比度值
        function updateContrast(app)
            app.contrastStrength = app.contrastSlider.Value;
            app.contrastValueLabel.Text = sprintf('当前值: %.2f', app.contrastStrength);
            app.updatePreview();
        end
        
        % 关闭对比度设置窗口
        function closeContrastSettings(app)
            delete(app.contrastSettingsFigure);
            app.contrastSettingsFigure = [];
        end
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            % 初始化画布显示
            app.imageHandle = imshow(app.imageData, 'Parent', app.UIAxes);
            app.UIAxes.XLim = [0.5 512.5];
            app.UIAxes.YLim = [0.5 512.5];
            app.updatePreview();
            
            % 设置默认数据集路径（数字识别）
            app.dataFolder = 'DataSets';
            app.trainFile = fullfile(app.dataFolder, 'emnist-digits-train.csv');
            app.testFile = fullfile(app.dataFolder, 'emnist-digits-test.csv');
            app.mappingFile = fullfile(app.dataFolder, 'emnist-digits-mapping.txt');
            
            % 初始禁用识别按钮
            app.RecognizeButton.Enable = 'off';
            app.RecognizeTestSetButton.Enable = 'off';
            
            % 尝试加载默认的预训练模型
            modelFile = fullfile(fileparts(mfilename('fullpath')), 'Models', 'CNN_Digits_Model.mat');
            if isfile(modelFile)
                try
                    app.loadModel(modelFile);
                catch ME
                    app.TextArea.Value = {'请加载模型'};
                    app.lastLineIsResult = true;
                end
            else
                app.TextArea.Value = {'请加载模型'};
                app.lastLineIsResult = true;
            end
        end

        % Window button down function: UIFigure
        function UIFigureWindowButtonDown(app, event)
        % 鼠标按下，开始绘制
            currentPoint = app.UIAxes.CurrentPoint;
            currentPoint = currentPoint(1, 1:2);
            
            % 检查鼠标是否在绘图区域内
            if currentPoint(1) >= app.UIAxes.XLim(1) && currentPoint(1) <= app.UIAxes.XLim(2) && currentPoint(2) >= app.UIAxes.YLim(1) && currentPoint(2) <= app.UIAxes.YLim(2)
                app.isDrawing = true;
                app.previousPoint = currentPoint;
            end
        end

        % Window button motion function: UIFigure
        function UIFigureWindowButtonMotion(app, event)
        % 鼠标移动，绘制轨迹
            if app.isDrawing
                currentPoint = app.UIAxes.CurrentPoint;
                currentPoint = currentPoint(1, 1:2);
                
                % 计算从前一点到当前点的插值点（确保连续线条）
                numPoints = round(norm(currentPoint - app.previousPoint)) + 1;
                xCoords = round(linspace(app.previousPoint(1), currentPoint(1), numPoints));
                yCoords = round(linspace(app.previousPoint(2), currentPoint(2), numPoints));
                
                % 过滤出有效坐标（在画布范围内）
                validIndices = find(xCoords >= 1 & xCoords <= 512 & yCoords >= 1 & yCoords <= 512);
                xCoords = xCoords(validIndices);
                yCoords = yCoords(validIndices);

                % 圆形笔刷
                for i = 1:length(xCoords)
                    x = xCoords(i);
                    y = yCoords(i);

                    x_start = max(1, x - app.brushRadius);
                    x_end = min(512, x + app.brushRadius);
                    y_start = max(1, y - app.brushRadius);
                    y_end = min(512, y + app.brushRadius);

                    for px = x_start:x_end
                        for py = y_start:y_end
                            if (px - x)^2 + (py - y)^2 <= app.brushRadius^2
                                app.imageData(py, px) = 0;
                            end
                        end
                    end
                end
                
                % 更新状态和显示
                app.previousPoint = currentPoint;
                app.imageHandle.CData = app.imageData;
                app.updatePreview();
            end
        end

        % Window button up function: UIFigure
        function UIFigureWindowButtonUp(app, event)
            % 结束绘制
            app.isDrawing = false;
            app.previousPoint = [];
        end

        % Button pushed function: ClearButton
        function ClearButtonPushed(app, event)
            % 重置画布为白色背景
            app.imageData = ones(512, 512);
            app.imageHandle.CData = app.imageData;
            app.updatePreview();
        end

        % Button pushed function: RecognizeButton
        function RecognizeButtonPushed(app, event)
            % 检查是否有可用模型
            if isempty(app.CNNModel) && isempty(app.rfModel)
                app.TextArea.Value = {'请先训练或加载一个模型！'};
                app.lastLineIsResult = true;
                return;
            end
            
            % 显示当前模型信息
            if ~isempty(app.CNNModel)
                modelInfo = sprintf('模型: CNN\n训练次数: %d\n准确率: %.2f%%', ...
                    app.trainingEpochs, app.modelAccuracy * 100);
            else
                modelInfo = sprintf('模型: Random Forest\n树数量: %d\n准确率: %.2f%%', ...
                    app.rfNumTrees, app.rfAccuracy * 100);
            end
            app.TextArea.Value = strsplit(modelInfo, '\n');
            app.lastLineIsResult = false;
            
            % 执行字符识别
            app.recognizeCharacter();
        end

        % Button pushed function: RecognizeTestSetButton
        function RecognizeTestSetButtonPushed(app, event)
            % 检查模型和数据文件
            if isempty(app.CNNModel) && isempty(app.rfModel)
                app.TextArea.Value = {'请先训练或加载一个模型！'};
                app.lastLineIsResult = true;
                return;
            end

            if ~isfile(app.trainFile) || ~isfile(app.testFile) || ~isfile(app.mappingFile)
                app.TextArea.Value = {'错误: 数据文件未找到！'};
                app.lastLineIsResult = true;
                return;
            end

            try
                app.TextArea.Value = {'正在加载测试数据...'};
                drawnow;
                
                % 读取标签映射和测试数据
                map_data = readmatrix(app.mappingFile);
                charLabels = char(map_data(:, 2));
                test_data_raw = readmatrix(app.testFile);
                YTest_raw = test_data_raw(:, 1);
                XTest_raw = test_data_raw(:, 2:end);

                % 随机选择8个测试样本
                numTestImages = size(XTest_raw, 1);
                randomIndices = randperm(numTestImages, 8);
                selectedImages = XTest_raw(randomIndices, :);
                selectedLabels = YTest_raw(randomIndices);
                
                % 准备图像数据
                XTest = zeros(28, 28, 1, 8, 'single');
                for i = 1:8
                    img = reshape(selectedImages(i, :), [28, 28]);
                    XTest(:, :, 1, i) = single(img);
                end
                XTest = XTest / 255;
                
                % 转换标签为字符
                switch app.ChooseSet.Value
                    case '数字'
                        YTest_char = charLabels(selectedLabels + 1);
                    case '字母'
                        YTest_char = charLabels(selectedLabels);
                    case '数字和字母'
                        YTest_char = charLabels(selectedLabels + 1);
                end
                trueLabels = cellstr(YTest_char);

                % 执行预测
                predictedLabels = cell(1, 8);
                if ~isempty(app.CNNModel)
                    img_batch = XTest;
                    predictedLabels = cellstr(classify(app.CNNModel, img_batch));
                elseif ~isempty(app.rfModel)
                    XTestVec = reshape(XTest, [], size(XTest,4))';
                    predictions = predict(app.rfModel, XTestVec);
                    predictedLabels = cellstr(predictions);
                end
                
                % 计算准确率
                correctCount = sum(strcmp(trueLabels, predictedLabels));
                accuracy = correctCount / 8 * 100;
                
                % 显示结果窗口
                createTestResultsFigure(app, XTest, trueLabels, predictedLabels, accuracy);
                
                % 更新主界面信息
                if ~isempty(app.CNNModel)
                    modelInfo = sprintf('模型: CNN\n训练次数: %d\n准确率: %.2f%%', ...
                        app.trainingEpochs, app.modelAccuracy * 100);
                else
                    modelInfo = sprintf('模型: Random Forest\n树数量: %d\n准确率: %.2f%%', ...
                        app.rfNumTrees, app.rfAccuracy * 100);
                end
                app.TextArea.Value = strsplit(modelInfo, '\n');
                app.lastLineIsResult = false;

                resultStr = sprintf('测试集识别完成，准确率: %.1f%% (%d/8)', accuracy, correctCount);
                currentContent = app.TextArea.Value;
                currentContent{end+1} = resultStr;
                app.TextArea.Value = currentContent;
                app.lastLineIsResult = true;

            catch ME
                app.TextArea.Value = {['测试集识别时发生错误: ' ME.message]};
                app.lastLineIsResult = true;
            end
        end

        % Value changed function: ChooseModel
        function ChooseModelValueChanged(app, event)
            % 忽略加载模型时的触发！！！
            if app.isLoadingModel
                return;
            end
            
            app.TextArea.Value = {'模型已更改，请重新训练或加载。'};
            app.lastLineIsResult = true;
            
            app.RecognizeButton.Enable = 'off';
            app.RecognizeTestSetButton.Enable = 'off';
        end

        % Value changed function: ChooseSet
        function ChooseSetValueChanged(app, event)
            % 根据选择更新数据文件路径
            switch app.ChooseSet.Value
                case '数字'
                    app.dataFolder = 'DataSets';
                    app.trainFile = fullfile(app.dataFolder, 'emnist-digits-train.csv');
                    app.testFile = fullfile(app.dataFolder, 'emnist-digits-test.csv');
                    app.mappingFile = fullfile(app.dataFolder, 'emnist-digits-mapping.txt');
                    title(app.UIAxes, '请在白色方框内绘制数字');
                case '字母'
                    app.dataFolder = 'DataSets';
                    app.trainFile = fullfile(app.dataFolder, 'emnist-letters-train.csv');
                    app.testFile = fullfile(app.dataFolder, 'emnist-letters-test.csv');
                    app.mappingFile = fullfile(app.dataFolder, 'emnist-letters-mapping.txt');
                    title(app.UIAxes, '请在白色方框内绘制字母');
                case '数字和字母'
                    app.dataFolder = 'DataSets';
                    app.trainFile = fullfile(app.dataFolder, 'emnist-balanced-train.csv');
                    app.testFile = fullfile(app.dataFolder, 'emnist-balanced-test.csv');
                    app.mappingFile = fullfile(app.dataFolder, 'emnist-balanced-mapping.txt');
                    title(app.UIAxes, '请在白色方框内绘制数字和字母');
            end
            
            % 清空当前模型并提示重新训练
            app.CNNModel = [];
            app.rfModel = [];
            app.TextArea.Value = {['数据集已切换为: ' app.ChooseSet.Value '，请重新训练模型。']};
            app.lastLineIsResult = true;
            
            % 禁用识别按钮
            app.RecognizeButton.Enable = 'off';
            app.RecognizeTestSetButton.Enable = 'off';
        end

        % Button pushed function: TrainButton
        function TrainButtonPushed(app, event)
            % 禁用按钮，防止重复操作
            app.TrainButton.Enable = 'off';
            app.LoadButton.Enable = 'off';
            app.TextArea.Value = {'正在准备数据，请稍候...'};
            drawnow;
            
            % 验证数据文件存在
            if ~isfile(app.trainFile) || ~isfile(app.testFile) || ~isfile(app.mappingFile)
                app.TextArea.Value = {'错误: 数据文件未找到！'};
                app.TrainButton.Enable = 'on';
                app.LoadButton.Enable = 'on';
                return;
            end

            try
                % 读取和准备训练数据
                map_data = readmatrix(app.mappingFile);
                charLabels = char(map_data(:, 2));
                train_data_raw = readmatrix(app.trainFile);
                test_data_raw = readmatrix(app.testFile);
                YTrain_raw = train_data_raw(:, 1);
                XTrain_raw = train_data_raw(:, 2:end);
                YTest_raw = test_data_raw(:, 1);
                XTest_raw = test_data_raw(:, 2:end);
                % 预分配图像数组
                numTrainImages = size(XTrain_raw, 1);
                numTestImages = size(XTest_raw, 1);
                XTrain = zeros(28, 28, 1, numTrainImages, 'single');
                XTest = zeros(28, 28, 1, numTestImages, 'single');
                % 重塑图像数据为28x28格式
                for i = 1:numTrainImages
                    img = reshape(XTrain_raw(i, :), [28, 28]);
                    XTrain(:, :, 1, i) = single(img);
                end
                for i = 1:numTestImages
                    img = reshape(XTest_raw(i, :), [28, 28]);
                    XTest(:, :, 1, i) = single(img);
                end

                % 归一化
                XTrain = XTrain / 255;
                XTest = XTest / 255;
                
                % 转换标签
                switch app.ChooseSet.Value
                    case '数字'
                        YTrain_char = charLabels(YTrain_raw + 1);
                        YTest_char = charLabels(YTest_raw + 1);
                    case '字母'
                        YTrain_char = charLabels(YTrain_raw);
                        YTest_char = charLabels(YTest_raw);
                    case '数字和字母'
                        YTrain_char = charLabels(YTrain_raw + 1);
                        YTest_char = charLabels(YTest_raw + 1);
                end
                YTrain = categorical(cellstr(YTrain_char));
                YTest = categorical(cellstr(YTest_char));
                
                % 根据选择的模型类型进行训练
                switch app.ChooseModel.Value
                    case 'CNN'
                        app.trainCNNModel(XTrain, YTrain, XTest, YTest);
                    case 'Random Forest'
                        app.trainRFModel(XTrain, YTrain, XTest, YTest);
                    otherwise
                        app.TextArea.Value = {'未知模型类型。'};
                end
            catch ME
                app.TextArea.Value = {['训练过程中发生错误: ' ME.message]};
            end
            
            % 重新启用按钮
            app.TrainButton.Enable = 'on';
            app.LoadButton.Enable = 'on';
        end

        % Button pushed function: LoadButton
        function LoadButtonPushed(app, event)
            % 防止重复操作
            app.TrainButton.Enable = 'off';
            app.LoadButton.Enable = 'off';
            app.TextArea.Value = {'请选择要加载的模型文件...'};
            drawnow;
            
            % 设置默认打开路径为Models文件夹
            appFilePath = mfilename('fullpath');
            [appFolder, ~, ~] = fileparts(appFilePath);
            modelsFolder = fullfile(appFolder, 'Models');
            if ~exist(modelsFolder, 'dir')
                mkdir(modelsFolder);
            end
            defaultPath = modelsFolder;

            % 打开文件选择对话框
            [file, path] = uigetfile('*.mat', '请选择模型文件 (.mat)',defaultPath);
            if file == 0
                app.TextArea.Value = {'已取消加载。'};
                app.lastLineIsResult = true;
                app.TrainButton.Enable = 'on';
                app.LoadButton.Enable = 'on';
                return;
            end
            % 加载模型
            app.loadModel(fullfile(path, file));
            
            % 重新按钮
            app.TrainButton.Enable = 'on';
            app.LoadButton.Enable = 'on';
        end

        % Button pushed function: ResetButton
        function ResetButtonPushed(app, event)
            % 重置画布
            app.imageData = ones(512, 512);
            app.imageHandle.CData = app.imageData;
            app.updatePreview();
            
            % 清空信息显示
            app.TextArea.Value = {''};
        
            app.RecognizeButton.Enable = 'off';
            app.RecognizeTestSetButton.Enable = 'off';
            
            % 重置模型状态
            app.CNNModel = [];
            app.modelAccuracy = 0;
            app.trainingEpochs = 0;
            app.ChooseModel.Value = 'CNN';
            app.lastLineIsResult = false;

            % 重置随机森林状态
            app.rfModel = [];
            app.rfAccuracy = 0;
            app.rfNumTrees = 100;

            % 重新启用训练和加载按钮
            app.TrainButton.Enable = 'on';
            app.LoadButton.Enable = 'on';
            
            % 尝试重新加载默认模型
            modelFile = fullfile(fileparts(mfilename('fullpath')), 'Models', 'CNN_Digits_Model.mat');
            if isfile(modelFile)
                try
                    app.loadModel(modelFile);
                catch ME
                    app.TextArea.Value = {'请加载模型'};
                    app.lastLineIsResult = true;
                end
            else
                app.TextArea.Value = {'请加载模型'};
                app.lastLineIsResult = true;
            end
        end

        % Selection changed function: ButtonGroup
        function ButtonGroupSelectionChanged(app, event)
            app.updatePreview();
        end

        % Menu selected function: PaintMenu
        function PaintMenuSelected(app, event)
            % 如果设置窗口已存在，则关闭
            if isfield(app, 'brushSettingsFigure') && ~isempty(app.brushSettingsFigure) && isvalid(app.brushSettingsFigure)
                delete(app.brushSettingsFigure);
                app.brushSettingsFigure = [];
                return;
            end
            
            % 创建笔刷设置窗口
            app.brushSettingsFigure = uifigure('Name', '笔刷设置', ...
                'WindowStyle', 'normal', ...
                'Position', [200 200 300 200], ...
                'CloseRequestFcn', @(src, event)closeBrushSettings(app));

            grid = uigridlayout(app.brushSettingsFigure, [4 1]);
            grid.RowHeight = {'1x', '1x', '1x', '1x'};
            grid.ColumnWidth = {'1x'};
            grid.RowSpacing = 5;
            grid.ColumnSpacing = 5;

            titleLabel = uilabel(grid);
            titleLabel.Layout.Row = 1;
            titleLabel.Layout.Column = 1;
            titleLabel.Text = '调整笔刷半径';
            titleLabel.FontSize = 14;
            titleLabel.FontWeight = 'bold';
            titleLabel.HorizontalAlignment = 'center';

            brushSlider = uislider(grid);
            brushSlider.Layout.Row = 2;
            brushSlider.Layout.Column = 1;
            brushSlider.Limits = [2 64];
            brushSlider.Value = app.brushRadius;
            brushSlider.ValueChangedFcn = @(src, event)updateBrushRadius(app, src);

            app.brushValueLabel = uilabel(grid);
            app.brushValueLabel.Layout.Row = 3;
            app.brushValueLabel.Layout.Column = 1;
            app.brushValueLabel.Text = sprintf('当前半径: %d', app.brushRadius);
            app.brushValueLabel.Tag = 'valueLabel';
            app.brushValueLabel.HorizontalAlignment = 'center';

            closeBtn = uibutton(grid, 'push');
            closeBtn.Layout.Row = 4;
            closeBtn.Layout.Column = 1;
            closeBtn.Text = '关闭';
            closeBtn.ButtonPushedFcn = @(src, event)closeBrushSettings(app);
        end

        % Menu selected function: ContrastMenu
        function ContrastMenuSelected(app, event)
            if ~isempty(app.contrastSettingsFigure) && isvalid(app.contrastSettingsFigure)
                delete(app.contrastSettingsFigure);
                app.contrastSettingsFigure = [];
                return;
            end

            % 对比度设置窗口
            app.contrastSettingsFigure = uifigure('Name', '对比度设置', ...
                'WindowStyle', 'normal', ...
                'Position', [200 200 300 200], ...
                'CloseRequestFcn', @(src, event)closeContrastSettings(app));

            grid = uigridlayout(app.contrastSettingsFigure, [4 1]);
            grid.RowHeight = {'1x', '1x', '1x', '1x'};
            grid.ColumnWidth = {'1x'};
            grid.RowSpacing = 5;
            grid.ColumnSpacing = 5;

            titleLabel = uilabel(grid);
            titleLabel.Layout.Row = 1;
            titleLabel.Layout.Column = 1;
            titleLabel.Text = '调整对比度增强值';
            titleLabel.FontSize = 14;
            titleLabel.FontWeight = 'bold';
            titleLabel.HorizontalAlignment = 'center';

            app.contrastValueLabel = uilabel(grid);
            app.contrastValueLabel.Layout.Row = 2;
            app.contrastValueLabel.Layout.Column = 1;
            app.contrastValueLabel.Text = sprintf('当前值: %.2f', app.contrastStrength);
            app.contrastValueLabel.HorizontalAlignment = 'center';
            app.contrastValueLabel.FontSize = 12;

            app.contrastSlider = uislider(grid);
            app.contrastSlider.Layout.Row = 3;
            app.contrastSlider.Layout.Column = 1;
            app.contrastSlider.Limits = [-1 1];
            app.contrastSlider.Value = app.contrastStrength;
            app.contrastSlider.ValueChangedFcn = @(src, event)updateContrast(app);

            closeBtn = uibutton(grid, 'push');
            closeBtn.Layout.Row = 4;
            closeBtn.Layout.Column = 1;
            closeBtn.Text = '关闭';
            closeBtn.ButtonPushedFcn = @(src, event)closeContrastSettings(app);
        end

        % Menu selected function: CNNMenu
        function CNNMenuSelected(app, event)
            if ~isempty(app.cnnSettingsFigure) && isvalid(app.cnnSettingsFigure)
                delete(app.cnnSettingsFigure);
                app.cnnSettingsFigure = [];
                return;
            end

            % CNN参数设置窗口
            app.cnnSettingsFigure = uifigure('Name', 'CNN参数设置', ...
                'WindowStyle', 'modal', ...
                'Position', [200 200 300 200], ...
                'CloseRequestFcn', @(src, event)delete(app.cnnSettingsFigure));

            grid = uigridlayout(app.cnnSettingsFigure, [4 2]);
            grid.RowHeight = {'1x', '1x', '1x', '1x'};
            grid.ColumnWidth = {'1x', '1x'};
            grid.RowSpacing = 5;
            grid.ColumnSpacing = 5;

            titleLabel = uilabel(grid);
            titleLabel.Layout.Row = 1;
            titleLabel.Layout.Column = [1 2];
            titleLabel.Text = 'CNN训练参数设置';
            titleLabel.FontSize = 14;
            titleLabel.FontWeight = 'bold';
            titleLabel.HorizontalAlignment = 'center';

            epochsLabel = uilabel(grid);
            epochsLabel.Layout.Row = 2;
            epochsLabel.Layout.Column = 1;
            epochsLabel.Text = '最大训练轮数:';
            epochsLabel.HorizontalAlignment = 'right';

            app.maxEpochsEdit = uieditfield(grid, 'numeric');
            app.maxEpochsEdit.Layout.Row = 2;
            app.maxEpochsEdit.Layout.Column = 2;
            app.maxEpochsEdit.Value = 15;
            app.maxEpochsEdit.Limits = [1 100];

            lrLabel = uilabel(grid);
            lrLabel.Layout.Row = 3;
            lrLabel.Layout.Column = 1;
            lrLabel.Text = '初始学习率:';
            lrLabel.HorizontalAlignment = 'right';

            app.initialLearnRateEdit = uieditfield(grid, 'numeric');
            app.initialLearnRateEdit.Layout.Row = 3;
            app.initialLearnRateEdit.Layout.Column = 2;
            app.initialLearnRateEdit.Value = 0.001; % 默认值
            app.initialLearnRateEdit.Limits = [0.0001 0.1]; % 设置范围

            applyBtn = uibutton(grid, 'push');
            applyBtn.Layout.Row = 4;
            applyBtn.Layout.Column = 1;
            applyBtn.Text = '应用';
            applyBtn.ButtonPushedFcn = @(src, event)applyCNNSettings(app);

            cancelBtn = uibutton(grid, 'push');
            cancelBtn.Layout.Row = 4;
            cancelBtn.Layout.Column = 2;
            cancelBtn.Text = '取消';
            cancelBtn.ButtonPushedFcn = @(src, event)delete(app.cnnSettingsFigure);
        end

        % Menu selected function: RandomForestMenu
        function RandomForestMenuSelected(app, event)
            if ~isempty(app.rfSettingsFigure) && isvalid(app.rfSettingsFigure)
                delete(app.rfSettingsFigure);
                app.rfSettingsFigure = [];
                return;
            end
            
            % 随机森林参数设置窗口
            app.rfSettingsFigure = uifigure('Name', '随机森林参数设置', ...
                'WindowStyle', 'modal', ...
                'Position', [200 200 300 200], ...
                'CloseRequestFcn', @(src, event)delete(app.rfSettingsFigure));

            grid = uigridlayout(app.rfSettingsFigure, [3 2]);
            grid.RowHeight = {'1x', '1x', '1x'};
            grid.ColumnWidth = {'1x', '1x'};
            grid.RowSpacing = 5;
            grid.ColumnSpacing = 5;

            titleLabel = uilabel(grid);
            titleLabel.Layout.Row = 1;
            titleLabel.Layout.Column = [1 2];
            titleLabel.Text = '随机森林参数设置';
            titleLabel.FontSize = 14;
            titleLabel.FontWeight = 'bold';
            titleLabel.HorizontalAlignment = 'center';

            treesLabel = uilabel(grid);
            treesLabel.Layout.Row = 2;
            treesLabel.Layout.Column = 1;
            treesLabel.Text = '树的数量:';
            treesLabel.HorizontalAlignment = 'right';

            app.numTreesEdit = uieditfield(grid, 'numeric');
            app.numTreesEdit.Layout.Row = 2;
            app.numTreesEdit.Layout.Column = 2;
            app.numTreesEdit.Value = app.rfNumTrees;
            app.numTreesEdit.Limits = [10 500];

            applyBtn = uibutton(grid, 'push');
            applyBtn.Layout.Row = 3;
            applyBtn.Layout.Column = 1;
            applyBtn.Text = '应用';
            applyBtn.ButtonPushedFcn = @(src, event)applyRFSettings(app);

            cancelBtn = uibutton(grid, 'push');
            cancelBtn.Layout.Row = 3;
            cancelBtn.Layout.Column = 2;
            cancelBtn.Text = '取消';
            cancelBtn.ButtonPushedFcn = @(src, event)delete(app.rfSettingsFigure);
        end

        % Menu selected function: AuthorRuixiMenu
        function AuthorRuixiMenuSelected(app, event)
            if ~isempty(app.AuthorFigure) && isvalid(app.AuthorFigure)
                delete(app.AuthorFigure);
                app.AuthorFigure = [];
                return;
            end
        
            app.AuthorFigure = uifigure('Name', '关于', ...
                'WindowStyle', 'normal', ...
                'Position', [200 200 300 370], ...
                'CloseRequestFcn', @(src, event) app.closeAboutWindow(src, event));
        
            grid = uigridlayout(app.AuthorFigure, [10 1]);
            grid.RowHeight = {'1.2x', '0.8x', '0.2x', '0.8x', '0.6x', '0.6x', '0.2x', '0.6x', '0.6x', '0.6x'}; % 增加一行高度
            grid.ColumnWidth = {'1x'};
            grid.RowSpacing = 5;
            grid.ColumnSpacing = 5;
            grid.Padding = [20 20 20 20];
        
            titleLabel = uilabel(grid);
            titleLabel.Layout.Row = 1;
            titleLabel.Layout.Column = 1;
            titleLabel.Text = 'Matlab/ML Demo';
            titleLabel.FontSize = 20;
            titleLabel.FontWeight = 'bold';
            titleLabel.HorizontalAlignment = 'center';
        
            versionLabel = uilabel(grid);
            versionLabel.Layout.Row = 2;
            versionLabel.Layout.Column = 1;
            versionLabel.Text = 'Version 2.3.0';
            versionLabel.FontSize = 12;
            versionLabel.FontColor = [0.3 0.3 0.3];
            versionLabel.HorizontalAlignment = 'center';
        
            authorLabel = uilabel(grid);
            authorLabel.Layout.Row = 4;
            authorLabel.Layout.Column = 1;
            authorLabel.Text = 'Author: Ruixi';
            authorLabel.FontSize = 14;
            authorLabel.HorizontalAlignment = 'center';
        
            emailLink = uilabel(grid);
            emailLink.Layout.Row = 5;
            emailLink.Layout.Column = 1;
            emailLink.Text = 'Email: Ruixi.Cheng@Outlook.com';
            emailLink.FontSize = 10;
            emailLink.HorizontalAlignment = 'center';
        
            githubLink = uilabel(grid);
            githubLink.Layout.Row = 6;
            githubLink.Layout.Column = 1;
            githubLink.Text = 'GitHub: github.com/Ruixi-Cheng'; 
            githubLink.FontSize = 10;
            githubLink.HorizontalAlignment = 'center';
        
            copyrightLabel = uilabel(grid);
            copyrightLabel.Layout.Row = 8;
            copyrightLabel.Layout.Column = 1;
            copyrightLabel.Text = 'Copyright © 2025 Ruixi';
            copyrightLabel.FontSize = 10;
            copyrightLabel.FontColor = [0.5 0.5 0.5];
            copyrightLabel.HorizontalAlignment = 'center';
        
            licenseLabel = uilabel(grid);
            licenseLabel.Layout.Row = 9;
            licenseLabel.Layout.Column = 1;
            licenseLabel.Text = 'SPDX-License-Identifier: LGPL-3.0';
            licenseLabel.FontSize = 10;
            licenseLabel.FontColor = [0.5 0.5 0.5];
            licenseLabel.HorizontalAlignment = 'center';
        
            editTimeLabel = uilabel(grid);
            editTimeLabel.Layout.Row = 10;
            editTimeLabel.Layout.Column = 1;
            editTimeLabel.Text = 'Last Edited: 2025/11/16';
            editTimeLabel.FontSize = 10;
            editTimeLabel.FontColor = [0.5 0.5 0.5];
            editTimeLabel.HorizontalAlignment = 'center';
        end

        function closeAboutWindow(app, src, ~)
            delete(src);
            app.AuthorFigure = [];
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.WindowButtonDownFcn = createCallbackFcn(app, @UIFigureWindowButtonDown, true);
            app.UIFigure.WindowButtonUpFcn = createCallbackFcn(app, @UIFigureWindowButtonUp, true);
            app.UIFigure.WindowButtonMotionFcn = createCallbackFcn(app, @UIFigureWindowButtonMotion, true);

            % Create Menu
            app.Menu = uimenu(app.UIFigure);
            app.Menu.Text = '参数设置';

            % Create PaintMenu
            app.PaintMenu = uimenu(app.Menu);
            app.PaintMenu.MenuSelectedFcn = createCallbackFcn(app, @PaintMenuSelected, true);
            app.PaintMenu.Text = '绘图';

            % Create ContrastMenu
            app.ContrastMenu = uimenu(app.Menu);
            app.ContrastMenu.MenuSelectedFcn = createCallbackFcn(app, @ContrastMenuSelected, true);
            app.ContrastMenu.Text = '对比度';

            % Create CNNMenu
            app.CNNMenu = uimenu(app.Menu);
            app.CNNMenu.MenuSelectedFcn = createCallbackFcn(app, @CNNMenuSelected, true);
            app.CNNMenu.Text = 'CNN';

            % Create RandomForestMenu
            app.RandomForestMenu = uimenu(app.Menu);
            app.RandomForestMenu.MenuSelectedFcn = createCallbackFcn(app, @RandomForestMenuSelected, true);
            app.RandomForestMenu.Text = 'Random Forest';

            % Create AuthorRuixiMenu
            app.AuthorRuixiMenu = uimenu(app.UIFigure);
            app.AuthorRuixiMenu.MenuSelectedFcn = createCallbackFcn(app, @AuthorRuixiMenuSelected, true);
            app.AuthorRuixiMenu.Text = 'Author: Ruixi';

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {'0.5x', '0.5x', '1x', '1x', '1x', '1x', '0.5x', '0.5x', '0.5x', '0.5x', '0.5x', '0.5x', '0.5x', '0.5x'};
            app.GridLayout.RowHeight = {'1x', '1x', '1x', '0.5x', '0.5x', '0.5x', '0.5x', '1x', '1x', '1x', '1x'};

            % Create UIAxes
            app.UIAxes = uiaxes(app.GridLayout);
            title(app.UIAxes, '请在白色方框内绘制数字')
            app.UIAxes.XLim = [0.5 512.5];
            app.UIAxes.YLim = [0.5 512.5];
            app.UIAxes.XColor = 'none';
            app.UIAxes.XTick = [];
            app.UIAxes.YColor = 'none';
            app.UIAxes.YTick = [];
            app.UIAxes.ZColor = 'none';
            app.UIAxes.Box = 'on';
            app.UIAxes.Layout.Row = [1 10];
            app.UIAxes.Layout.Column = [1 8];

            % Create PreviewAxes
            app.PreviewAxes = uiaxes(app.GridLayout);
            title(app.PreviewAxes, '模型输入预览')
            app.PreviewAxes.XColor = 'none';
            app.PreviewAxes.XTick = [];
            app.PreviewAxes.YColor = 'none';
            app.PreviewAxes.YTick = [];
            app.PreviewAxes.ZColor = 'none';
            app.PreviewAxes.Layout.Row = [1 4];
            app.PreviewAxes.Layout.Column = [9 14];

            % Create ClearButton
            app.ClearButton = uibutton(app.GridLayout, 'push');
            app.ClearButton.ButtonPushedFcn = createCallbackFcn(app, @ClearButtonPushed, true);
            app.ClearButton.Layout.Row = 11;
            app.ClearButton.Layout.Column = [6 8];
            app.ClearButton.Text = '清除绘图';

            % Create RecognizeButton
            app.RecognizeButton = uibutton(app.GridLayout, 'push');
            app.RecognizeButton.ButtonPushedFcn = createCallbackFcn(app, @RecognizeButtonPushed, true);
            app.RecognizeButton.Layout.Row = 11;
            app.RecognizeButton.Layout.Column = [9 11];
            app.RecognizeButton.Text = '识别绘图';

            % Create TextArea
            app.TextArea = uitextarea(app.GridLayout);
            app.TextArea.Editable = 'off';
            app.TextArea.Layout.Row = [9 10];
            app.TextArea.Layout.Column = [9 14];

            % Create Label
            app.Label = uilabel(app.GridLayout);
            app.Label.HorizontalAlignment = 'right';
            app.Label.Layout.Row = 6;
            app.Label.Layout.Column = [9 11];
            app.Label.Text = '选择模型';

            % Create ChooseModel
            app.ChooseModel = uidropdown(app.GridLayout);
            app.ChooseModel.Items = {'CNN', 'Random Forest'};
            app.ChooseModel.ValueChangedFcn = createCallbackFcn(app, @ChooseModelValueChanged, true);
            app.ChooseModel.Layout.Row = 6;
            app.ChooseModel.Layout.Column = [12 14];
            app.ChooseModel.Value = 'CNN';

            % Create TrainButton
            app.TrainButton = uibutton(app.GridLayout, 'push');
            app.TrainButton.ButtonPushedFcn = createCallbackFcn(app, @TrainButtonPushed, true);
            app.TrainButton.Layout.Row = 8;
            app.TrainButton.Layout.Column = [9 11];
            app.TrainButton.Text = '训练';

            % Create ResetButton
            app.ResetButton = uibutton(app.GridLayout, 'push');
            app.ResetButton.ButtonPushedFcn = createCallbackFcn(app, @ResetButtonPushed, true);
            app.ResetButton.Layout.Row = 11;
            app.ResetButton.Layout.Column = [1 2];
            app.ResetButton.Text = '重置软件';

            % Create LoadButton
            app.LoadButton = uibutton(app.GridLayout, 'push');
            app.LoadButton.ButtonPushedFcn = createCallbackFcn(app, @LoadButtonPushed, true);
            app.LoadButton.Layout.Row = 8;
            app.LoadButton.Layout.Column = [12 14];
            app.LoadButton.Text = '加载';

            % Create ButtonGroup
            app.ButtonGroup = uibuttongroup(app.GridLayout);
            app.ButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @ButtonGroupSelectionChanged, true);
            app.ButtonGroup.BorderWidth = 0;
            app.ButtonGroup.Layout.Row = 5;
            app.ButtonGroup.Layout.Column = [9 14];

            % Create OriginalImage
            app.OriginalImage = uiradiobutton(app.ButtonGroup);
            app.OriginalImage.Text = '原图像';
            app.OriginalImage.Position = [1 2 67 19];
            app.OriginalImage.Value = true;

            % Create ContrastEnhancement
            app.ContrastEnhancement = uiradiobutton(app.ButtonGroup);
            app.ContrastEnhancement.Text = '对比度增强';
            app.ContrastEnhancement.Position = [67 -10 82 42];

            % Create Binarization
            app.Binarization = uiradiobutton(app.ButtonGroup);
            app.Binarization.Text = '二值化';
            app.Binarization.Position = [157 -12 67 44];

            % Create Label_2
            app.Label_2 = uilabel(app.GridLayout);
            app.Label_2.HorizontalAlignment = 'right';
            app.Label_2.Layout.Row = 7;
            app.Label_2.Layout.Column = [9 11];
            app.Label_2.Text = '选择识别对象';

            % Create ChooseSet
            app.ChooseSet = uidropdown(app.GridLayout);
            app.ChooseSet.Items = {'数字', '字母', '数字和字母'};
            app.ChooseSet.ValueChangedFcn = createCallbackFcn(app, @ChooseSetValueChanged, true);
            app.ChooseSet.Layout.Row = 7;
            app.ChooseSet.Layout.Column = [12 14];
            app.ChooseSet.Value = '数字';

            % Create RecognizeTestSetButton
            app.RecognizeTestSetButton = uibutton(app.GridLayout, 'push');
            app.RecognizeTestSetButton.ButtonPushedFcn = createCallbackFcn(app, @RecognizeTestSetButtonPushed, true);
            app.RecognizeTestSetButton.Layout.Row = 11;
            app.RecognizeTestSetButton.Layout.Column = [12 14];
            app.RecognizeTestSetButton.Text = '识别测试集';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Recognizer_exp

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end