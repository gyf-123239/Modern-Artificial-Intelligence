# 实验五————多模态情感分析

## 项目简介

本项目旨在解决基于配对的文本和图像数据进行情感分类的任务，情感标签分为三类：positive、neutral 和 negative。为了更好地利用文本和图像这两种信息，本项目设计了一个多模态融合模型，将两种模态的信息进行有效融合，提升情感分类的准确性。

## 关键技术

1. **多模态融合模型**：该模型结合了文本模型（BERT）和图像模型（ResNet50），通过提取文本和图像的特征，进行联合学习和分类。  
  
2. **数据预处理与增强**：文本数据通过BERT的分词器进行处理，图像数据使用标准的图像增强技术（如旋转、翻转等）进行处理，增加数据的多样性，提升模型的泛化能力。  
  
3. **标签编码**：使用`LabelEncoder`对情感标签进行编码，从而方便模型训练与评估。  
  
4. **早停法**：训练过程中采用早停策略，在验证集上出现连续若干次不提升时，提前停止训练，防止过拟合。  
  
5. **调整超参数**：模型训练中包括多个超参数的调整，如学习率、批大小、dropout率和权重衰减等。本项目通过这些超参数的优化，进一步提升模型性能并控制了过拟合。

## 实验环境

- Python版本：使用的Python版本为 3.9.21，并通过 conda 虚拟环境进行管理。  
  
- CUDA版本：使用的CUDA版本为 11.5.2,cuDNN版本为8.3.2，支持GPU加速。  
  
- PyTorch版本：使用的是 PyTorch GPU版本，确保支持CUDA。  
  
运行以下代码下载相关python库：  

```python
pip install -r requirements.txt
```

## 项目结构

```python
|-- data5                      数据集文件夹
    |-- data                       包含训练文本和图片，每个文件按照唯一的guid命名。
    |-- test_without_label.txt     没有标签的测试数据
    |-- train.txt                  训练数据
|-- multimodal.py              项目代码
|-- test_predictions.txt       测试集结果文件（运行multimodal.py生成）
|-- requirements.txt           需要安装的python包
|-- README.md                  项目说明文件
```

## 运行程序

- 运行 multimodal.py

```python
python -u "文件路径（例如F:\code\multimodal.py）"
```

## 结果输出

- 训练时每个轮次打印训练集损失，训练集准确率，验证集损失，验证集准确率
- 训练早停时，打印早停的训练轮次，以及最佳模型的验证准确率
- 绘制训练集和验证集的loss下降曲线图
- 生成测试集结果文件test_predictions.txt  

## 消融实验

```python
    # 融合
    model = initialize_model(text_model_name=text_model_name, image_model_name=image_model_name, dropout_rate=dropout_rate)
    # 仅使用文本数据
    # model_text_only = initialize_model(text_model_name=text_model_name, image_model_name=None, num_classes=3, use_text=True, use_image=False, dropout_rate=dropout_rate)
    # 仅使用图像数据
    # model_image_only = initialize_model(text_model_name=text_model_name, image_model_name=None, num_classes=3, use_text=False, use_image=True, dropout_rate=dropout_rate)
```

```python
    # 融合
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 仅文本
    # optimizer = torch.optim.AdamW(model_text_only.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 仅图像
    # optimizer = torch.optim.AdamW(model_image_only.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

```python
    # 融合
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience)
    # 仅文本
    # train_model(model_text_only, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience)  
    # 仅图像
    # train_model(model_image_only, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience)  
```

在main函数中的上述代码段中，可以选择不同的数据输入，包括融合，仅文本，仅图像，更换注释，然后直接运行multimodal.py文件即可。会输出消融实验的结果。  
