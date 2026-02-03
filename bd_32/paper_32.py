import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

DATASET_BIN = 'dataset_all.bin'
MATRIX_SIZE = 32
INPUT_DIM = 32 * 32
BRANCH_MIN = 2
BRANCH_MAX = 12
NUM_CLASSES = BRANCH_MAX - BRANCH_MIN + 1  # 11 classes
TARGET_PER_CLASS = 10000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_dataset_all(path):
    """
    Load dataset from binary file
    Each sample:
      uint8 bd
      uint8[1024] matrix
    """
    raw = np.fromfile(path, dtype=np.uint8)

    record_size = 1 + INPUT_DIM
    if raw.size % record_size != 0:
        raise ValueError("Invalid dataset_all.bin file size")

    num = raw.size // record_size
    print(f"Detected samples: {num}")

    X = []
    y = []

    offset = 0
    for _ in range(num):
        bd = int(raw[offset])
        offset += 1

        mat = raw[offset:offset + INPUT_DIM]
        offset += INPUT_DIM

        if bd < BRANCH_MIN or bd > BRANCH_MAX:
            continue

        X.append(mat.reshape(32, 32))
        y.append(bd)

    return np.array(X, dtype=np.int8), np.array(y, dtype=np.int32)


def augment_matrix(m):
    """
    Apply random row/column operations (GF(2) equivalence)
    """
    m = m.copy()
    ops = random.randint(1, 4)

    for _ in range(ops):
        op = random.choice(['rswap', 'cswap', 'rxor', 'cxor'])

        if op == 'rswap':
            i, j = random.sample(range(32), 2)
            m[[i, j]] = m[[j, i]]

        elif op == 'cswap':
            i, j = random.sample(range(32), 2)
            m[:, [i, j]] = m[:, [j, i]]

        elif op == 'rxor':
            i, j = random.sample(range(32), 2)
            m[i] ^= m[j]

        elif op == 'cxor':
            i, j = random.sample(range(32), 2)
            m[:, i] ^= m[:, j]

    return m


def expand_dataset(X, y):
    """
    Expand dataset to target number of samples per class
    """
    X_out = []
    y_out = []

    for bd in range(BRANCH_MIN, BRANCH_MAX + 1):
        mats = X[y == bd]
        label = bd - BRANCH_MIN

        # Bd=12: Use original samples without augmentation
        if bd == 12:
            for m in mats:
                X_out.append(m)
                y_out.append(label)
            print(f"Bd=12 using original {len(mats)} samples")
            continue

        # Bd=2~11: Expand to target per class
        need = TARGET_PER_CLASS
        idx = 0
        while len([v for v in y_out if v == label]) < need:
            base = mats[idx % len(mats)]
            if idx < len(mats):
                X_out.append(base)
            else:
                X_out.append(augment_matrix(base))
            y_out.append(label)
            idx += 1

        print(f"Bd={bd} expanded to {need} samples")

    return np.array(X_out, dtype=np.int8), np.array(y_out, dtype=np.int32)


def create_enhanced_model():
    """
    Create enhanced neural network model for branch classification
    """
    model = keras.Sequential([
        layers.Dense(1024, activation='swish', input_shape=(INPUT_DIM,),
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(512, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),

        layers.Dense(256, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(NUM_CLASSES, activation='softmax')
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
            filepath='best_branch_classifier_32x32.h5',
            monitor='val_prc',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            'training_log_32x32.csv',
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
    plt.savefig('training_history_32x32.png', dpi=300)
    plt.show()


def evaluate_model(model, X_test, y_test_onehot, y_test_int):
    """
    Evaluate model performance with comprehensive metrics
    """
    print("\nEvaluating model performance...")
    results = model.evaluate(X_test, y_test_onehot, verbose=0)

    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'auc': results[4],
        'prc': results[5]
    }

    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test PRC: {metrics['prc']:.4f}")

    # Calculate F1 Score
    f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-7)
    print(f"Test F1 Score: {f1:.4f}")

    # Classification Report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    target_names = [f'Branch {i + BRANCH_MIN}' for i in range(NUM_CLASSES)]
    print(f"\nClassification Report:")
    print(classification_report(y_test_int, y_pred_classes, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test_int, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig('confusion_matrix_32x32.png', dpi=300)
    plt.show()

    return metrics


def train_model(X, y):
    """
    Train enhanced neural network model for branch classification
    """
    print("Preparing training data...")

    X = X.reshape(X.shape[0], -1)
    print("Label distribution:", np.bincount(y))

    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    y_train_onehot = keras.utils.to_categorical(y_train_int, NUM_CLASSES)
    y_test_onehot = keras.utils.to_categorical(y_test_int, NUM_CLASSES)

    print("\nCreating enhanced model...")
    model = create_enhanced_model()
    model.summary()

    callbacks = get_enhanced_callbacks()

    print("\nStarting model training...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train_onehot,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    duration = time.time() - start_time
    print(f"Model training completed, time: {duration / 60:.2f} minutes")

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test_onehot, y_test_int)

    # Plot training history
    plot_training_history(history)

    # Save final model
    model.save('branch_classifier_32x32_enhanced.h5')
    print("Model saved as 'branch_classifier_32x32_enhanced.h5'")

    # Prediction examples
    print("\nPrediction Examples:")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for i, idx in enumerate(sample_indices):
        sample_matrix = X_test[idx].reshape(1, -1)
        true_label = y_test_int[idx] + BRANCH_MIN
        pred = model.predict(sample_matrix, verbose=0)
        pred_label = np.argmax(pred) + BRANCH_MIN
        confidence = np.max(pred)
        print(
            f"Example {i + 1}: True branch: {true_label}, Predicted branch: {pred_label}, Confidence: {confidence:.4f}")
        print(f"  Probability distribution: {[f'{p:.4f}' for p in pred[0]]}")

    return model, history


if __name__ == '__main__':
    # Load raw dataset
    X_raw, y_raw = load_dataset_all(DATASET_BIN)

    print("Original data distribution:")
    for bd in range(2, 13):
        print(f"  Bd={bd}: {(y_raw == bd).sum()}")

    # Expand dataset
    X, y = expand_dataset(X_raw, y_raw)

    print(f"\nTotal samples after expansion: {len(X)}")

    # Train enhanced model
    model, history = train_model(X, y)