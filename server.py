from flask import Flask, abort, request, jsonify, render_template
from paddle import fluid
import numpy as np
import csv
import scipy.integrate as spi
import numpy as np
import math
TRAIN_FILE = './data/deaths/England_2021-02-27.csv'
TEST_DATA = []
app = Flask(__name__)


@app.route('/')
def index():
    data = {}
    with open("./COVID-19-master/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv", 'r') as f:
        reader = list(csv.reader(f))
        length = len(reader) - 1
        title = list(reader[0][4:len(reader[0])])
        content = list(reader[length][4:len(reader[length])])
        data['c1_title'] = title
        data['c1_content'] = content
        f.close()
    with open("./COVID-19-master/archived_data/archived_time_series/time_series_19-covid-Recovered_archived_0325.csv", 'r') as f2:
        reader2 = list(csv.reader(f2))
        length = len(reader2) - 1
        title2 = list(reader2[0][4:len(reader2[0])])
        content2 = list(reader2[length][4:len(reader2[length])])
        data['c2_title'] = title2
        data['c2_content'] = content2
        f2.close()
    with open("./COVID-19-master/archived_data/archived_time_series/time_series_19-covid-Deaths_archived_0325.csv", 'r') as f3:
        reader3 = list(csv.reader(f3))
        length = len(reader3) - 1
        title3 = list(reader3[0][4:len(reader3[0])])
        content3 = list(reader3[length][4:len(reader3[length])])
        data['c3_title'] = title3
        data['c3_content'] = content3
        f3.close()
    return render_template('index.html', **data)

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

def data_pretreatment():
    global TEST_DATA
    TEST_DATA = []
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
        for i in range(train_arr_length - 31*2, train_arr_length - 31):
            temp = []
            temp_row = train_arr[i: i + 31]
            for item in temp_row:
                if is_number(item):
                    temp.append(normalization(float(item),max_value,0))
                else:
                    temp.append(0.0)

            row_data = np.array(temp, dtype='float32')
            TEST_DATA.append(row_data[:30])
        return max_value, train_arr




@app.route('/dnn_infer/', methods=['GET'])
def dnn_infer():
    max_value, real_data = data_pretreatment()
    # place = fluid.CUDAPlace(0)
    place = fluid.CPUPlace()
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    pred_array = []
    # finall_pred_array = []
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model("./model", infer_exe)
        iris_feat = np.array(TEST_DATA).astype("float32")
        assert feed_target_names[0] == 'input'
        results = infer_exe.run(inference_program,
                                feed={feed_target_names[0]: np.array(iris_feat)},
                                fetch_list=fetch_targets)
        # print("Iris results: (Iris Type)")
        for idx, val in enumerate(results[0]):
            pred = round(anti_normalization(val[0], max_value, 0))
            # pred = val[0]
            pred_array.append(pred)
            # print("%d: %f" % (idx, pred))
        temp_list = real_data[0: len(real_data) - 31]
        # for i in range(len(temp_list)):
        #     temp_list[i] = round(anti_normalization(temp_list[i], max_value, 0))
        temp_list.extend(pred_array)
        finall_pred_array = temp_list
    data = {}
    data['infer'] = finall_pred_array
    data['x'] = range(len(real_data))
    data['real'] = real_data
    print("real_data:",len(real_data))
    # print(real_data)
    print("temp_list:", len(finall_pred_array))
    # print(finall_pred_array)
    # exit()
    return render_template('dnn.html', **data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8383, debug=True)
