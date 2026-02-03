import os
import ast
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, \
    average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ========== Step 1: Load Original Data ==========
root_dir = './chushishujuji'
matrices = []
labels = []

print("Loading original data...")

for branch_num in range(1, 6):
    subdir = f'bn{branch_num}'
    subdir_path = os.path.join(root_dir, subdir)

    if not os.path.exists(subdir_path):
        print(f"Warning: Directory {subdir_path} does not exist")
        continue

    for file_name in os.listdir(subdir_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(subdir_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            try:
                matrix_list = ast.literal_eval(content)
                matrix = np.array(matrix_list)
                if matrix.shape == (8, 8):
                    matrices.append(matrix)
                    labels.append(branch_num)
                else:
                    print(f"Warning: {file_path} is not 8x8")
            except Exception as e:
                print(f"Error: Failed to parse {file_path}: {e}")

print(f"Successfully loaded {len(matrices)} original matrices")

# Group by label
matrices_by_label = {}
for m, l in zip(matrices, labels):
    matrices_by_label.setdefault(l, []).append(m)


# ========== Step 2: Data Augmentation ==========
def apply_row_operations(matrix):
    new_matrix = matrix.copy()
    operations = random.sample(['swap', 'flip', 'xor'], k=random.randint(1, 3))
    for op in operations:
        if op == 'swap':
            i, j = random.sample(range(8), 2)
            new_matrix[[i, j]] = new_matrix[[j, i]]
        elif op == 'flip':
            row = random.randint(0, 7)
            new_matrix[row] = 1 - new_matrix[row]
        elif op == 'xor':
            i, j = random.sample(range(8), 2)
            new_matrix[i] = np.bitwise_xor(new_matrix[i], new_matrix[j])
    return new_matrix


augmented_matrices = []
augmented_labels = []

for label, matrices_list in matrices_by_label.items():
    print(f"Processing branch {label}...")
    original_count = len(matrices_list)
    variants_per_matrix = 10000 // original_count
    extra_needed = 10000 % original_count

    for i, matrix in enumerate(matrices_list):
        augmented_matrices.append(matrix)
        augmented_labels.append(label)
        variants_to_generate = variants_per_matrix - 1
        if i < extra_needed:
            variants_to_generate += 1
        for _ in range(variants_to_generate):
            variant = apply_row_operations(matrix)
            augmented_matrices.append(variant)
            augmented_labels.append(label)

    print(f"Branch {label} expanded to {len([l for l in augmented_labels if l == label])}")

augmented_matrices = np.array(augmented_matrices)
augmented_labels = np.array(augmented_labels)

print(f"Augmented dataset: {augmented_matrices.shape}")
print(f"Label distribution: {np.bincount(augmented_labels)}")


# ========== Step 3: Enhanced Model Architecture ==========
def create_enhanced_model():
    model = keras.Sequential([
        layers.Dense(256, activation='swish', input_shape=(64,),
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(64, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.15),

        layers.Dense(4, activation='softmax')
    ])

    # Focal Loss for imbalanced data
    def focal_loss(gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            modulating_factor = tf.pow(1.0 - p_t, gamma)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
            loss = alpha_factor * modulating_factor * cross_entropy
            return tf.reduce_mean(loss)

        return loss

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision', top_k=1),
            keras.metrics.Recall(name='recall', top_k=1),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR')
        ]
    )
    return model


# ========== Step 4: Enhanced Callbacks ==========
def get_enhanced_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_prc',
            patience=15,
            verbose=1,
            mode='max',
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_prc',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='best_branch_classifier.h5',
            monitor='val_prc',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            'training_log_branch.csv',
            separator=',',
            append=False
        )
    ]


# ========== Step 5: Enhanced Visualization ==========
def plot_training_history(history):
    plt.figure(figsize=(18, 12))

    # Accuracy curve
    plt.subplot(2, 3, 1)
    plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()

    # Loss curve
    plt.subplot(2, 3, 2)
    plt.plot(history.history['loss'], 'b-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Model Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()

    # Precision-Recall curve
    plt.subplot(2, 3, 3)
    plt.plot(history.history['precision'], 'g-', label='Training Precision')
    plt.plot(history.history['val_precision'], 'm-', label='Validation Precision')
    plt.plot(history.history['recall'], 'c-', label='Training Recall')
    plt.plot(history.history['val_recall'], 'y-', label='Validation Recall')
    plt.title('Precision and Recall', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()

    # AUC curve
    plt.subplot(2, 3, 4)
    plt.plot(history.history['auc'], 'b-', label='Training AUC')
    plt.plot(history.history['val_auc'], 'r-', label='Validation AUC')
    plt.title('AUC Curve', fontsize=14)
    plt.ylabel('AUC', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()

    # PR curve
    plt.subplot(2, 3, 5)
    plt.plot(history.history['prc'], 'b-', label='Training PRC')
    plt.plot(history.history['val_prc'], 'r-', label='Validation PRC')
    plt.title('PR Curve', fontsize=14)
    plt.ylabel('PRC', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_branch.png', dpi=300)
    plt.show()


# ========== Step 6: Enhanced Evaluation ==========
def evaluate_model(model, X_test, y_test_onehot, y_test_int, history=None):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)

    # 获取训练和验证的最终指标
    if history is not None:
        # 获取最佳epoch（早停恢复的权重对应的epoch）
        best_epoch = np.argmax(history.history['val_prc'])

        print(f"\n{'=' * 30} Training Metrics {'=' * 30}")
        print(f"Best Epoch: {best_epoch + 1}")
        print(f"Training Accuracy (final epoch): {history.history['accuracy'][-1]:.4f}")
        print(f"Training Accuracy (best epoch): {history.history['accuracy'][best_epoch]:.4f}")
        print(f"Validation Accuracy (final epoch): {history.history['val_accuracy'][-1]:.4f}")
        print(f"Validation Accuracy (best epoch): {history.history['val_accuracy'][best_epoch]:.4f}")
        print(f"Training Loss (final epoch): {history.history['loss'][-1]:.4f}")
        print(f"Training Loss (best epoch): {history.history['loss'][best_epoch]:.4f}")
        print(f"Validation Loss (final epoch): {history.history['val_loss'][-1]:.4f}")
        print(f"Validation Loss (best epoch): {history.history['val_loss'][best_epoch]:.4f}")

        # 保存训练和验证指标
        train_val_metrics = {
            'training_accuracy_final': history.history['accuracy'][-1],
            'training_accuracy_best': history.history['accuracy'][best_epoch],
            'validation_accuracy_final': history.history['val_accuracy'][-1],
            'validation_accuracy_best': history.history['val_accuracy'][best_epoch],
            'training_loss_final': history.history['loss'][-1],
            'training_loss_best': history.history['loss'][best_epoch],
            'validation_loss_final': history.history['val_loss'][-1],
            'validation_loss_best': history.history['val_loss'][best_epoch]
        }
    else:
        train_val_metrics = {}

    print(f"\n{'=' * 30} Test Set Metrics {'=' * 30}")
    # 模型评估
    results = model.evaluate(X_test, y_test_onehot, verbose=0)

    metrics = {
        'test_loss': results[0],
        'test_accuracy': results[1],
        'test_precision': results[2],
        'test_recall': results[3],
        'test_auc': results[4],
        'test_prc': results[5]
    }

    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Test Recall: {metrics['test_recall']:.4f}")
    print(f"Test AUC (ROC): {metrics['test_auc']:.4f}")
    print(f"Test PRC (AUC-PR): {metrics['test_prc']:.4f}")

    # 计算F1 Score
    f1 = 2 * (metrics['test_precision'] * metrics['test_recall']) / (
                metrics['test_precision'] + metrics['test_recall'] + 1e-7)
    metrics['test_f1'] = f1
    print(f"Test F1 Score: {f1:.4f}")

    # 预测
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_test_int, y_pred_classes, average=None
    )

    # 计算宏平均和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test_int, y_pred_classes, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test_int, y_pred_classes, average='weighted'
    )

    print(f"\n{'=' * 30} Per-Class Metrics {'=' * 30}")
    for i, label_name in enumerate(['Branch 2', 'Branch 3', 'Branch 4', 'Branch 5']):
        print(
            f"{label_name}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")

    print(f"\nMacro Average: Precision={precision_macro:.4f}, Recall={recall_macro:.4f}, F1={f1_macro:.4f}")
    print(f"Weighted Average: Precision={precision_weighted:.4f}, Recall={recall_weighted:.4f}, F1={f1_weighted:.4f}")

    # 分类报告
    print(f"\n{'=' * 30} Classification Report {'=' * 30}")
    print(classification_report(y_test_int, y_pred_classes,
                                target_names=['Branch 2', 'Branch 3', 'Branch 4', 'Branch 5']))

    # 混淆矩阵
    cm = confusion_matrix(y_test_int, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Branch 2', 'Branch 3', 'Branch 4', 'Branch 5'],
                yticklabels=['Branch 2', 'Branch 3', 'Branch 4', 'Branch 5'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig('confusion_matrix_branch.png', dpi=300)
    plt.show()

    # 合并所有指标
    all_metrics = {**train_val_metrics, **metrics}
    all_metrics.update({
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    })

    return all_metrics


# ========== Step 7: Prepare Data for Training ==========
# Use only branches 2-5, map to 0-3
mask = augmented_labels >= 2
X = augmented_matrices[mask]
y = augmented_labels[mask] - 2  # Convert labels to 0-based

print(f"Classification dataset size: {X.shape}, Label distribution: {np.bincount(y)}")

# Reshape to flat vectors
X_flat = X.reshape(X.shape[0], -1)

# Train/Test split
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode labels for training
num_classes = 4
y_train_onehot = keras.utils.to_categorical(y_train_int, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test_int, num_classes)

# ========== Step 8: Create and Train Model ==========
print("\nCreating enhanced model...")
model = create_enhanced_model()
model.summary()

callbacks = get_enhanced_callbacks()

print("\nStarting model training...")
start_time = time.time()

history = model.fit(
    X_train, y_train_onehot,
    batch_size=64,
    epochs=100,
    validation_split=0.15,
    verbose=1,
    callbacks=callbacks
)

duration = time.time() - start_time
print(f"Model training completed, time: {duration / 60:.2f} minutes")

# ========== Step 9: Evaluate Model ==========
metrics = evaluate_model(model, X_test, y_test_onehot, y_test_int, history)

# ========== Step 10: Visualize Training History ==========
plot_training_history(history)

# ========== Step 11: Save Metrics to CSV ==========
# 创建指标汇总表
metrics_summary = {
    'Metric': [
        'Training Accuracy (final epoch)',
        'Training Accuracy (best epoch)',
        'Validation Accuracy (final epoch)',
        'Validation Accuracy (best epoch)',
        'Training Loss (final epoch)',
        'Training Loss (best epoch)',
        'Validation Loss (final epoch)',
        'Validation Loss (best epoch)',
        'Test Accuracy',
        'Test Loss',
        'Test Precision',
        'Test Recall',
        'Test F1 Score',
        'Test AUC (ROC)',
        'Test PRC (AUC-PR)',
        'Precision (Macro)',
        'Recall (Macro)',
        'F1 Score (Macro)',
        'Precision (Weighted)',
        'Recall (Weighted)',
        'F1 Score (Weighted)'
    ],
    'Value': [
        metrics.get('training_accuracy_final', np.nan),
        metrics.get('training_accuracy_best', np.nan),
        metrics.get('validation_accuracy_final', np.nan),
        metrics.get('validation_accuracy_best', np.nan),
        metrics.get('training_loss_final', np.nan),
        metrics.get('training_loss_best', np.nan),
        metrics.get('validation_loss_final', np.nan),
        metrics.get('validation_loss_best', np.nan),
        metrics.get('test_accuracy', np.nan),
        metrics.get('test_loss', np.nan),
        metrics.get('test_precision', np.nan),
        metrics.get('test_recall', np.nan),
        metrics.get('test_f1', np.nan),
        metrics.get('test_auc', np.nan),
        metrics.get('test_prc', np.nan),
        metrics.get('precision_macro', np.nan),
        metrics.get('recall_macro', np.nan),
        metrics.get('f1_macro', np.nan),
        metrics.get('precision_weighted', np.nan),
        metrics.get('recall_weighted', np.nan),
        metrics.get('f1_weighted', np.nan)
    ]
}

df_metrics = pd.DataFrame(metrics_summary)
df_metrics.to_csv('model_metrics_summary.csv', index=False)
print(f"\nMetrics summary saved to 'model_metrics_summary.csv'")

# 打印所有指标的汇总表
print(f"\n{'=' * 60}")
print("ALL METRICS SUMMARY")
print('=' * 60)
print(df_metrics.to_string(index=False))

# ========== Step 12: Save Final Model ==========
model.save('matrix_branch_classifier_enhanced.h5')
print("\nModel saved as 'matrix_branch_classifier_enhanced.h5'")

# ========== Step 13: Prediction Examples ==========
print(f"\n{'=' * 30} Prediction Examples {'=' * 30}")
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for i, idx in enumerate(sample_indices):
    sample_matrix = X_test[idx].reshape(1, -1)
    true_label = y_test_int[idx] + 2
    pred = model.predict(sample_matrix, verbose=0)
    pred_label = np.argmax(pred) + 2
    confidence = np.max(pred)
    print(f"Example {i + 1}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.4f}")
    print(f"Probability Distribution: {pred[0]}")