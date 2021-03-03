from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import math
import csv
from paddle.utils.plot import Ploter
import sys

# 数据初始化
TRAIN_DATA = []
TEST_DATA = []
TRAIN_RES = []
TEST_RES = []
TRAIN_FILE = './data/deaths/England_2021-02-27.csv'
TEST_FILE = ''

def normalization(x, max, min):
    x = (x - min) / (max - min)
    return x

def anti_normalization(x, max, min):
    x = x * (max - min) + min
    return x

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# 数据预处理
def data_pretreatment():
    # train
    with open(TRAIN_FILE) as f:
        # 数据预处理
        render = csv.reader(f)
        max_value = 0.0
        train_arr = []
        for row in render:
            if row[0] != '' and is_number(row[4]):
                train_arr.append(float(row[4]))
                if float(row[4]) > max_value:
                    max_value = float(row[4])
        train_arr.reverse()
        # 预处理结束
        train_arr_length = len(train_arr)
        for i in range(train_arr_length):
            if i > train_arr_length - 31:
                break
            temp = []
            temp_row = train_arr[i: i + 31]
            for item in temp_row:
                if is_number(item):
                    temp.append(normalization(float(item),max_value,0))
                    # temp.append(item)
                else:
                    temp.append(0.0)

            row_data = np.array(temp, dtype='float32')
            TRAIN_DATA.append(row_data[:30])
            TRAIN_RES.append([row_data[30]])
        for i in range(train_arr_length - 31*2, train_arr_length - 31):
            temp = []
            temp_row = train_arr[i: i + 31]
            for item in temp_row:
                if is_number(item):
                    temp.append(normalization(float(item),max_value,0))
                    # temp.append(item)
                else:
                    temp.append(0.0)

            row_data = np.array(temp, dtype='float32')
            TEST_DATA.append(row_data[:30])
            TEST_RES.append([row_data[30]])

data_pretreatment()
# print(TRAIN_DATA)
# print(TRAIN_RES)
# exit()
# for i in TRAIN_DATA:
#     print(i)
# # for i in TRAIN_RES:
# #     print(i)
# # print('---------------------')
# # for i in TEST_DATA:
# #     print(i)
# # for i in TEST_RES:
# #     print(i)
# exit()

# 训练集读取器
# TODO
def train_sample_reader():
    for i in range(120):
        input = np.array(TRAIN_DATA).astype('float32')
        output = np.array(TRAIN_RES).astype('float32')
        yield input, output

# 测试集读取器
def test_sample_reader():
    for i in range(30):
        input = np.array(TEST_DATA).astype('float32')
        output = np.array(TEST_RES).astype('float32')
        yield input, output

# 训练函数
def train(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]  # 累加测试过程中的损失值
        count += 1  # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated]

# 训练部分

# data pretreatment
# data_pretreatment()

# exit()


train_reader = fluid.io.batch(train_sample_reader, batch_size=100)
test_reader = fluid.io.batch(test_sample_reader, batch_size=100)

# network structure 网络结构
# the input is year, month and day
INPUT = fluid.data(name='input', shape=[None, 30], dtype='float32')
# the output is the number
OUTPUT = fluid.data(name='label', shape=[None, 1], dtype='float32')
hidden = fluid.layers.fc(name='fc1', input=INPUT, size=120, act='relu')
hidden = fluid.layers.fc(name='fc2', input=hidden, size=80, act='relu')
hidden = fluid.layers.fc(name='fc3', input=hidden, size=60, act='relu')
hidden = fluid.layers.fc(name='fc4', input=hidden, size=40, act='relu')
hidden = fluid.layers.fc(name='fc5', input=hidden, size=30, act='relu')
prediction = fluid.layers.fc(name='output', input=hidden, size=1, act='relu')

# main program
main_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()

# loss 损失函数
loss = fluid.layers.mean(fluid.layers.mse_loss(input=prediction, label=OUTPUT))

# test program
test_program = main_program.clone(for_test=True)
# 优化器使用Adam
adam = fluid.optimizer.Adam(learning_rate=0.001)
# 求最小损失
adam.minimize(loss)

# 选择gpu 0
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 迭代次数
num_epochs = 100

# 模型路径
params_dirname = "./model"
# datafeeder 数据读取器
feeder = fluid.DataFeeder(place=place, feed_list=[INPUT, OUTPUT])
exe.run(startup_program)

step = 0
# 测试用Executor
exe_test = fluid.Executor(place)
# 开始训练
for pass_id in range(num_epochs):
    # 读取数据
    for data_train in train_reader():
        step = step + 1
        # Executor开始加载
        avg_loss_value, = exe.run(main_program, feed=feeder.feed(data_train), fetch_list=[loss])
        # 够十次输出一次结果
        if step % 10 == 0:
            # 调用训练函数训练
            test_metics = train(executor=exe_test, program=test_program, reader=test_reader, fetch_list=[loss.name], feeder=feeder)
            # 反馈步长，损失
            print("%s, Step %d, Cost %f" % ("test cost", step, test_metics[0]))
            print("%s, Step %d, Cost %f" % ("train cost", step, avg_loss_value[0]))
            if test_metics[0] < 10.0:
                break
        # if cant training
        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")


        if params_dirname is not None:
            # params: model path, input layer name, output layer, Executor
            fluid.io.save_inference_model(params_dirname, ['input'], [prediction], exe)