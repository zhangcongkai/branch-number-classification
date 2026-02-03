import os
import re
import ast
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

INPUT_FOLDER = '16'
OUTPUT_DIR = './augmented_dataset_16x16'
VARIANTS_PER_ORIGINAL = 100
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def parse_assertions(content):
    """
    Parse ASSERT statements and return (varname, hex_value) list
    Pattern example: ASSERT(x100 = 0x1F);
    """
    pattern = r'ASSERT\(\s*(x\d+)\s*=\s*(0x[0-9A-Fa-f]+)\s*\);'
    matches = re.findall(pattern, content)
    return matches


def hex_to_binary_matrix(assertions):
    """
    Convert 16 ASSERT (xN, 0xVAL) statements to 16x16 binary matrix
    Sort by variable number (x100, x101, ...), each hex value expanded to 16-bit binary row
    """
    sorted_assertions = sorted(assertions, key=lambda x: int(x[0][1:]))

    matrix = []
    for var, hex_val in sorted_assertions:
        bin_val = bin(int(hex_val, 16))[2:].zfill(16)
        row = [int(bit) for bit in bin_val]
        matrix.append(row)

    return np.array(matrix, dtype=np.int8)


def process_files(folder_path):
    """
    Process all txt files in folder, extract each query (split by 'stp 1.cvc' or 'Invalid.'),
    and construct 16x16 matrices from queries with 16 ASSERT statements
    Return matrices (list of np.array) and labels (list of int)
    """
    matrices = []
    labels = []

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for file_name in files:
        label_match = re.search(r'bn(\d+)', file_name)
        if not label_match:
            continue
        label = int(label_match.group(1))

        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            if 'stp 1.cvc' in content:
                queries = re.split(r'stp 1\.cvc', content)
            else:
                queries = re.split(r'Invalid\.', content)

            for query in queries:
                if 'ASSERT' in query:
                    assertions = parse_assertions(query)
                    if len(assertions) == 16:
                        matrix = hex_to_binary_matrix(assertions)
                        matrices.append(matrix)
                        labels.append(label)

    return matrices, labels


def apply_row_operations(matrix):
    """
    Apply random row operations (swap / flip / xor) to 16x16 matrix
    Randomly select 1-3 operations (non-repeating) and apply in order
    """
    new_matrix = matrix.copy()
    operations = random.sample(['swap', 'flip', 'xor'], k=random.randint(1, 3))

    for op in operations:
        if op == 'swap' and len(new_matrix) >= 2:
            i, j = random.sample(range(16), 2)
            new_matrix[[i, j]] = new_matrix[[j, i]]

        elif op == 'flip':
            row = random.randint(0, 15)
            new_matrix[row] = 1 - new_matrix[row]

        elif op == 'xor' and len(new_matrix) >= 2:
            i, j = random.sample(range(16), 2)
            new_matrix[i] = np.bitwise_xor(new_matrix[i], new_matrix[j])

    return new_matrix


def augment_and_save(input_folder=INPUT_FOLDER, output_dir=OUTPUT_DIR,
                     variants_per_original=VARIANTS_PER_ORIGINAL):
    print("Loading original data...")
    matrices, labels = process_files(input_folder)
    print(f"Successfully loaded {len(matrices)} original matrices")

    matrices_by_label = {}
    for matrix, label in zip(matrices, labels):
        matrices_by_label.setdefault(label, []).append(matrix)

    augmented_matrices = []
    augmented_labels = []

    print("Starting data augmentation...")
    for label, matrices_list in matrices_by_label.items():
        print(f"  Processing label bn{label}, original samples: {len(matrices_list)}")
        for matrix in matrices_list:
            augmented_matrices.append(matrix)
            augmented_labels.append(label)

            for _ in range(variants_per_original - 1):
                variant = apply_row_operations(matrix)
                augmented_matrices.append(variant)
                augmented_labels.append(label)

        count_label = sum(1 for l in augmented_labels if l == label)
        print(f"    Label bn{label} augmented samples: {count_label}")

    augmented_matrices = np.array(augmented_matrices)
    augmented_labels = np.array(augmented_labels)

    print(f"Total augmented samples: {len(augmented_matrices)}")
    unique_labels = sorted(set(augmented_labels.tolist()))
    print("Sample counts per label:")
    for lab in unique_labels:
        print(f"  bn{lab}: {np.sum(augmented_labels == lab)}")

    os.makedirs(output_dir, exist_ok=True)
    for lab in unique_labels:
        indices = np.where(augmented_labels == lab)[0]
        lab_dir = os.path.join(output_dir, f'bn{lab}')
        os.makedirs(lab_dir, exist_ok=True)
        for i, idx in enumerate(indices):
            matrix = augmented_matrices[idx]
            file_path = os.path.join(lab_dir, f'matrix_{i:05d}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(matrix.tolist()))
        print(f"Saved {len(indices)} matrices for label bn{lab} to {lab_dir}")

    print("Data augmentation and saving completed!")
    return augmented_matrices, augmented_labels


def load_augmented_data(data_dir=OUTPUT_DIR):
    matrices = []
    labels = []

    for branch_num in range(2, 9):
        subdir = f'bn{branch_num}'
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Warning: Directory {subdir_path} does not exist, skipping")
            continue

        for file_name in os.listdir(subdir_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(subdir_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                try:
                    matrix_list = ast.literal_eval(content)
                    matrix = np.array(matrix_list, dtype=np.int8)
                    if matrix.shape == (16, 16):
                        matrices.append(matrix)
                        labels.append(branch_num)
                    else:
                        print(f"Warning: File {file_path} is not 16x16, actual shape {matrix.shape}")
                except (ValueError, SyntaxError) as e:
                    print(f"Error: Failed to parse {file_path}: {e}")

    return np.array(matrices), np.array(labels)


def create_enhanced_model(num_classes=7):
    """
    Create enhanced neural network model for branch classification
    """
    model = keras.Sequential([
        layers.Dense(512, activation='swish', input_shape=(256,),
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),

        layers.Dense(128, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(num_classes, activation='softmax')
    ])

    # Focal Loss for handling class imbalance
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


def get_enhanced_callbacks():
    """
    Create enhanced callbacks for model training
    """
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
            filepath='best_branch_classifier_16x16.h5',
            monitor='val_prc',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            'training_log_16x16.csv',
            separator=',',
            append=False
        )
    ]


def plot_training_history(history):
    """
    Plot comprehensive training history with multiple metrics
    """
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
    plt.savefig('training_history_16x16.png', dpi=300)
    plt.show()


def evaluate_model(model, X_test, y_test_onehot, y_test_int, num_classes=7, history=None):
    """
    Evaluate model performance with comprehensive metrics
    """
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
    target_names = [f'Branch {i + 2}' for i in range(num_classes)]
    for i, label_name in enumerate(target_names):
        print(
            f"{label_name}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")

    print(f"\nMacro Average: Precision={precision_macro:.4f}, Recall={recall_macro:.4f}, F1={f1_macro:.4f}")
    print(f"Weighted Average: Precision={precision_weighted:.4f}, Recall={recall_weighted:.4f}, F1={f1_weighted:.4f}")

    # 分类报告
    print(f"\n{'=' * 30} Classification Report {'=' * 30}")
    print(classification_report(y_test_int, y_pred_classes, target_names=target_names))

    # 混淆矩阵
    cm = confusion_matrix(y_test_int, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig('confusion_matrix_16x16.png', dpi=300)
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


def train_model(X, y, num_classes=7, output_model_path='matrix_branch_classifier_16x16_enhanced.h5'):
    """
    Train enhanced neural network model for branch classification
    """
    print("Preparing training data...")

    y = y - 2
    print(f"Dataset size: {X.shape}, Label distribution: {np.bincount(y)}")

    X_flat = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X_flat, y, test_size=0.2, random_state=SEED, stratify=y
    )

    y_train_onehot = keras.utils.to_categorical(y_train_int, num_classes)
    y_test_onehot = keras.utils.to_categorical(y_test_int, num_classes)

    print("\nCreating enhanced model...")
    model = create_enhanced_model(num_classes=num_classes)
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

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test_onehot, y_test_int, num_classes, history)

    # Plot training history
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
    df_metrics.to_csv('model_metrics_summary_16x16.csv', index=False)
    print(f"\nMetrics summary saved to 'model_metrics_summary_16x16.csv'")

    # 打印所有指标的汇总表
    print(f"\n{'=' * 60}")
    print("ALL METRICS SUMMARY")
    print('=' * 60)
    print(df_metrics.to_string(index=False))

    # Save final model
    model.save(output_model_path)
    print(f"\nModel saved as '{output_model_path}'")

    # Prediction examples
    print(f"\n{'=' * 30} Prediction Examples {'=' * 30}")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for i, idx in enumerate(sample_indices):
        sample_matrix = X_test[idx].reshape(1, -1)
        true_label = y_test_int[idx] + 2
        pred = model.predict(sample_matrix, verbose=0)
        pred_label = np.argmax(pred) + 2
        confidence = np.max(pred)
        print(
            f"Example {i + 1}: True branch: {true_label}, Predicted branch: {pred_label}, Confidence: {confidence:.4f}")
        print(f"  Probability distribution: {[f'{p:.4f}' for p in pred[0]]}")

    return model, history


if __name__ == "__main__":
    # First augment and save data
    augmented_matrices, augmented_labels = augment_and_save(
        input_folder=INPUT_FOLDER,
        output_dir=OUTPUT_DIR,
        variants_per_original=VARIANTS_PER_ORIGINAL
    )

    # Load augmented data
    X, y = load_augmented_data(data_dir=OUTPUT_DIR)

    if len(X) == 0:
        raise RuntimeError("No training data found. Please check the augmentation path or original data parsing.")

    # Train enhanced model (7 classes: bn2..bn8)
    model, history = train_model(X, y, num_classes=7,
                                 output_model_path='matrix_branch_classifier_16x16_enhanced.h5')