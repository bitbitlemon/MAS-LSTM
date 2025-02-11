import numpy as np
import os
import time  # 导入time模块
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from utils import load_data  # 假设这个文件包含数据加载的函数

# 设置随机种子和基本参数
seed = 42
DATA_PATH = "./v1_datasets/"
BATCH_SIZE = 128

# 加载数据集
cids = []
for _, _, cid in os.walk(DATA_PATH):
    cids.extend(cid)

silos = {}

for cid in cids:
    _cid = cid[:cid.find(".csv")]
    silos[_cid] = {}
    x_train, y_train, x_test, y_test = load_data.load_data(os.path.join(DATA_PATH, cid), info=False)

    # 直接使用原始训练数据和测试数据（不添加提示特征）
    silos[_cid]["x_train"] = x_train
    silos[_cid]["y_train"] = y_train
    silos[_cid]["x_test"] = x_test
    silos[_cid]["y_test"] = y_test


# 定义 LSTM 模型
def build_lstm(input_dim):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(input_dim, 1), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # 二分类问题
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 创建一个智能体类，包含训练和预测方法
class Agent:
    def __init__(self, input_dim):
        self.model = build_lstm(input_dim)

    def train(self, x_train, y_train):
        x_train = np.expand_dims(x_train, axis=-1)  # LSTM 需要三维输入 (samples, timesteps, features)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # 早停机制
        start_time = time.time()  # 记录训练开始时间
        self.model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE, verbose=0, validation_split=0.2,
                       callbacks=[early_stopping])
        end_time = time.time()  # 记录训练结束时间
        training_time = end_time - start_time  # 计算训练时间
        return training_time

    def predict(self, x_test):
        x_test = np.expand_dims(x_test, axis=-1)
        start_time = time.time()  # 记录预测开始时间
        predictions = (self.model.predict(x_test) > 0.5).astype(int)  # 返回二分类标签（0或1）
        end_time = time.time()  # 记录预测结束时间
        prediction_time = end_time - start_time  # 计算预测时间
        return predictions, prediction_time


# 创建多智能体系统
class MultiAgentSystem:
    def __init__(self, agents):
        self.agents = agents

    def train_agents(self, silos):
        total_training_time = 0
        for silo_name, silo_data in silos.items():
            agent = self.agents[silo_name]
            training_time = agent.train(silo_data['x_train'], silo_data['y_train'])
            total_training_time += training_time
        return total_training_time

    def evaluate_agents(self, silos):
        results = []
        total_prediction_time = 0
        for silo_name, silo_data in silos.items():
            agent = self.agents[silo_name]
            pred, prediction_time = agent.predict(silo_data['x_test'])
            # 评估性能
            results.append(self.evaluate_metrics(silo_data['y_test'], pred))
            total_prediction_time += prediction_time
        return results, total_prediction_time

    def evaluate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, pre, rec, f1


# 创建多智能体系统
agents = {silo_name: Agent(input_dim=silo_data['x_train'].shape[1]) for silo_name, silo_data in silos.items()}
multi_agent_system = MultiAgentSystem(agents)

# 训练和评估
total_training_time = multi_agent_system.train_agents(silos)
results, total_prediction_time = multi_agent_system.evaluate_agents(silos)

# 计算每个样本的训练时间和推理时间
avg_training_time_per_sample = total_training_time / sum([silo_data['x_train'].shape[0] for silo_data in silos.values()])
avg_prediction_time_per_sample = total_prediction_time / sum([silo_data['x_test'].shape[0] for silo_data in silos.values()])

# 输出结果并保存到txt文件
with open("v1_results.txt", "w") as f:
    for silo_name, (acc, pre, rec, f1) in zip(silos.keys(), results):
        f.write(f"Silo: {silo_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {pre:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write("\n")

    # 输出平均结果
    avg_acc = np.mean([r[0] for r in results])
    avg_pre = np.mean([r[1] for r in results])
    avg_rec = np.mean([r[2] for r in results])
    avg_f1 = np.mean([r[3] for r in results])

    f.write(f"Average Accuracy: {avg_acc:.4f}\n")
    f.write(f"Average Precision: {avg_pre:.4f}\n")
    f.write(f"Average Recall: {avg_rec:.4f}\n")
    f.write(f"Average F1-score: {avg_f1:.4f}\n")

    # 输出训练时间和预测时间
    f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
    f.write(f"Total Prediction Time: {total_prediction_time:.2f} seconds\n")
    f.write(f"Average Training Time per Sample: {avg_training_time_per_sample:.6f} seconds\n")
    f.write(f"Average Prediction Time per Sample: {avg_prediction_time_per_sample:.6f} seconds\n")
    print(f"Average F1-score: {avg_f1:.4f}\n")
    print(f"Average Accuracy: {avg_acc:.4f}\n")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Total Prediction Time: {total_prediction_time:.2f} seconds")
    print(f"Average Training Time per Sample: {avg_training_time_per_sample:.6f} seconds")
    print(f"Average Prediction Time per Sample: {avg_prediction_time_per_sample:.6f} seconds")

print("Evaluation results and time metrics saved to 'v1_results.txt'")
