import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model("best_branch_classifier_32x32.h5", compile=False)

n = 32
num_samples = 1000
matrices = np.random.randint(0, 2, size=(num_samples, n, n), dtype=np.uint8)

# 将矩阵展平为64维向量
X = matrices.reshape(num_samples, -1).astype(np.float32)

# 预热
_ = model.predict(X[:10], verbose=0)

start = time.perf_counter()
preds = model.predict(X, batch_size=256, verbose=0)
elapsed = time.perf_counter() - start
avg_ms = elapsed / num_samples * 1000
print(f"神经网络 (n={n}, 11分类): {elapsed:.4f} sec, {avg_ms:.3f} ms/matrix")