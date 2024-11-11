import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 数据预处理：将图像扩展为 28x28x1（灰度图像的通道数为1），并进行标准化
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 将标签转换为 one-hot 编码
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 创建CNN模型
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout to reduce overfitting
    keras.layers.Dense(10, activation='softmax')
])

# 编译模型，设置一个初始的学习率
initial_lr = 0.001
optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

# 编译模型
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型总结
model.summary()

# 训练CNN模型
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 评估模型
train_accuracy = history.history['accuracy'][-1] * 100
test_accuracy = history.history['val_accuracy'][-1] * 100

print(f'Training Accuracy: {train_accuracy:.2f}%')
print(f'Testing Accuracy: {test_accuracy:.2f}%')

# 输出训练损失率和验证损失率
print("Training Loss over epochs:", history.history['loss'])
print("Validation Loss over epochs:", history.history['val_loss'])

# 绘制训练和验证准确率的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 绘制训练和验证损失函数的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

new_lr = 0.0005  # 修改学习率
new_batch_size = 128  # 修改批次大小

# 更新优化器
new_optimizer = keras.optimizers.Adam(learning_rate=new_lr)

# 重新编译模型
model.compile(optimizer=new_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 重新训练模型
history_updated = model.fit(x_train, y_train, epochs=5, batch_size=new_batch_size, validation_data=(x_test, y_test))

# 评估模型
train_accuracy_updated = history_updated.history['accuracy'][-1] * 100
test_accuracy_updated = history_updated.history['val_accuracy'][-1] * 100

print(f'Updated Training Accuracy: {train_accuracy_updated:.2f}%')
print(f'Updated Testing Accuracy: {test_accuracy_updated:.2f}%')

# 输出训练损失率和验证损失率
print("Updated Training Loss over epochs:", history_updated.history['loss'])
print("Updated Validation Loss over epochs:", history_updated.history['val_loss'])

# 绘制训练和验证准确率的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history_updated.history['accuracy'], label='Updated Training Accuracy')
plt.plot(history_updated.history['val_accuracy'], label='Updated Validation Accuracy')
plt.title('Updated Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 绘制训练和验证损失函数的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history_updated.history['loss'], label='Updated Training Loss')
plt.plot(history_updated.history['val_loss'], label='Updated Validation Loss')
plt.title('Updated Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = model.predict(x_test)

# 将预测结果转换为类别
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

# 找到正确分类的索引
correct_idx = np.where(y_pred_class == y_test_class)[0]

# 随机选取5张正确分类的图片
sample_correct = np.random.choice(correct_idx, 5)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(sample_correct):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {y_pred_class[idx]} Actual: {y_test_class[idx]}')
    plt.axis('off')
plt.suptitle('Correctly Classified Images', fontsize=16)
plt.show()

incorrect_idx = np.where(y_pred_class != y_test_class)[0]

# 随机选取5张错误分类的图片
sample_incorrect = np.random.choice(incorrect_idx, 5)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(sample_incorrect):
    plt.subplot(2, 5, i+6)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {y_pred_class[idx]} Actual: {y_test_class[idx]}')
    plt.axis('off')
plt.suptitle('Incorrectly Classified Images', fontsize=16)
plt.show()


for idx in sample_incorrect:
    print(f"Incorrectly classified image (Pred: {y_pred_class[idx]}, Actual: {y_test_class[idx]})")
    print(f"Image shape: {x_test[idx].shape}")
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {y_pred_class[idx]} Actual: {y_test_class[idx]}')
    plt.show()

    # 分析错误原因
    if y_pred_class[idx] == 1 and y_test_class[idx] == 7:
        print("Potential reason: The model may confuse 1 with 7 due to similarities in some handwriting styles.")
    elif y_pred_class[idx] == 3 and y_test_class[idx] == 5:
        print("Potential reason: The model may confuse 3 with 5 because of the similarity in their shapes in certain writing styles.")
    else:
        print("Potential reason: Could be due to blurry images, noise, or overlapping strokes.")
    print("="*50)