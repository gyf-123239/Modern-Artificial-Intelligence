import torch
import torch.nn as nn
import transformers
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 定义多模态数据集类，用于存储和处理多模态数据
class MultimodalDataset(Dataset):
    # 读取文本数据文件、存储相关信息、创建标签编码器等初始化操作
    def __init__(self, text_file, image_dir, tokenizer, transform, max_length=512):
        self.text_data = pd.read_csv(text_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.text_data['encoded_label'] = self.label_encoder.fit_transform(self.text_data['tag'])

    # 返回数据集的长度
    def __len__(self):
        return len(self.text_data)

    # 根据索引 idx 获取数据集中的样本
    def __getitem__(self, idx):
        guid = self.text_data.iloc[idx, 0] 
        text_path = os.path.join(self.image_dir, f"{guid}.txt") 
        image_path = os.path.join(self.image_dir, f"{guid}.jpg")  

        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()

        image = Image.open(image_path).convert("RGB")

        label = self.text_data.loc[idx, 'encoded_label']

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        image = self.transform(image)

        return encoding, image, label


# 定义多模态融合模型
class MultimodalModel(nn.Module):
    # 初始化多模态模型，为前向传播做准备
    def __init__(self, text_model, image_model, num_classes=3, use_text=True, use_image=True, dropout_rate=0.5):
        super(MultimodalModel, self).__init__()

        self.text_model = text_model if use_text else None
        self.image_model = image_model if use_image else None

        input_dim = 768 if use_text else 0 
        input_dim += 2048 if use_image else 0 
        self.fc = nn.Linear(input_dim, num_classes) 
        self.dropout = nn.Dropout(dropout_rate)  

    # 定义模型的前向传播过程
    def forward(self, input_ids=None, attention_mask=None, image=None):
        features = []

        if self.text_model and input_ids is not None:
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state[:, 0, :] 
            features.append(text_features)

        if self.image_model and image is not None:
            image_features = self.image_model(image)
            features.append(image_features)

        combined_features = torch.cat(features, dim=1)
        combined_features = self.dropout(combined_features)  
        logits = self.fc(combined_features)
        return logits



# 数据预处理函数，进行数据集的划分，创建训练集和验证集，创建 DataLoader 用于批量加载数据
def load_and_preprocess_data(text_file, image_dir, tokenizer, image_transform, batch_size=16, max_length=512):
    dataset = MultimodalDataset(text_file, image_dir, tokenizer, image_transform, max_length)

    train_data, val_data = train_test_split(dataset.text_data, test_size=0.2, random_state=42) 

    train_dataset = MultimodalDataset(text_file, image_dir, tokenizer, image_transform, max_length)
    train_dataset.text_data = train_data.reset_index(drop=True)  

    val_dataset = MultimodalDataset(text_file, image_dir, tokenizer, image_transform, max_length)
    val_dataset.text_data = val_data.reset_index(drop=True) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# 模型初始化函数，加载预训练的 BertModel 和 ResNet50，移除 ResNet50 的最后一层分类头，创建 MultimodalModel 实例
def initialize_model(text_model_name='bert-base-uncased', image_model_name='resnet50', num_classes=3, dropout_rate=0.5, use_text=True, use_image=True):
    text_model = transformers.BertModel.from_pretrained(text_model_name)
    image_model = models.resnet50(pretrained=True)
    image_model.fc = nn.Identity()
    model = MultimodalModel(text_model, image_model, num_classes, use_text, use_image, dropout_rate)
    return model


# 训练函数，使用早停法进行模型的训练
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', patience=3):
    model.to(device)
    best_val_loss = float('inf')
    best_val_acc = 0  
    best_epoch = 0  
    patience_counter = 0
    model.train()

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练过程
        train_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            encoding, images, labels = batch
            input_ids = encoding['input_ids'].squeeze(1).to(device)
            attention_mask = encoding['attention_mask'].squeeze(1).to(device)
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = correct / total

        # 验证过程
        val_loss = 0
        val_correct = 0
        val_total = 0
        model.eval()

        with torch.no_grad():
            for batch in val_loader:
                encoding, images, labels = batch
                input_ids = encoding['input_ids'].squeeze(1).to(device)
                attention_mask = encoding['attention_mask'].squeeze(1).to(device)
                images = images.to(device)
                labels = labels.to(device).long()

                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 记录损失数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 早停法
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc 
            best_epoch = epoch + 1  
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping... Best epoch: {best_epoch}, Best Val Accuracy: {best_val_acc:.4f}")
                break

        model.train()

    plot_loss_curve(train_losses, val_losses)



# 绘制loss曲线图
def plot_loss_curve(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


# 预测函数，输出测试集的预测结果
def predict(model, test_data, tokenizer, image_transform, label_encoder, device='cuda'):
    model.to(device)
    model.eval()
    predictions = []

    for guid in test_data['guid']:
        image_path = os.path.join('data5/data/', f"{guid}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = image_transform(image).unsqueeze(0).to(device)

        encoding = tokenizer("", truncation=True, padding="max_length", max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, image)
            label = torch.argmax(output, dim=1).item()
            label_str = label_encoder.inverse_transform([label])[0]
            predictions.append(label_str)

    return predictions


# 保存预测结果，输出测试集结果文件
def save_predictions(predictions, test_file='data5/test_without_label.txt', output_file='test_predictions.txt'):
    test_data = pd.read_csv(test_file)

    test_data['tag'] = predictions

    with open(output_file, 'w') as f:
        f.write("guid,tag\n") 
        for idx, row in test_data.iterrows():
            f.write(f"{row['guid']},{row['tag']}\n")


# 主函数，调度整个实验过程
def main():
    # 超参数
    text_file = 'data5/train.txt' 
    image_dir = 'data5/data/' 
    test_file = 'data5/test_without_label.txt' 
    output_file = 'test_predictions.txt'  
    text_model_name = 'bert-base-uncased' 
    image_model_name = 'resnet50'  
    batch_size = 8 
    num_epochs = 20  
    learning_rate = 1e-6  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    max_length = 256  
    patience = 4 
    dropout_rate = 0.1 
    weight_decay = 0.01 

    tokenizer = transformers.BertTokenizer.from_pretrained(text_model_name)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    # 加载数据
    dataset = MultimodalDataset(text_file, image_dir, tokenizer, image_transform, max_length)
    label_encoder = dataset.label_encoder
    train_loader, val_loader = load_and_preprocess_data(text_file, image_dir, tokenizer, image_transform, batch_size, max_length)

    # 初始化模型
    model = initialize_model(text_model_name=text_model_name, image_model_name=image_model_name, dropout_rate=dropout_rate)
    # 仅使用文本数据
    # model_text_only = initialize_model(text_model_name=text_model_name, image_model_name=None, num_classes=3, use_text=True, use_image=False, dropout_rate=dropout_rate)
    # 仅使用图像数据
    # model_image_only = initialize_model(text_model_name=text_model_name, image_model_name=None, num_classes=3, use_text=False, use_image=True, dropout_rate=dropout_rate)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 仅文本
    # optimizer = torch.optim.AdamW(model_text_only.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 仅图像
    # optimizer = torch.optim.AdamW(model_image_only.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience)
    # 仅文本
    # train_model(model_text_only, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience)  
    # 仅图像
    # train_model(model_image_only, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience)  

    # 预测测试文件
    test_data = pd.read_csv(test_file)
    predictions = predict(model, test_data, tokenizer, image_transform, label_encoder, device)
    save_predictions(predictions, test_file, output_file)

if __name__ == '__main__':
    main()