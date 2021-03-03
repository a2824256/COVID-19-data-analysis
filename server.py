from flask import Flask, abort, request, jsonify, render_template
from paddle import fluid
import numpy as np
import csv
import scipy.integrate as spi
import numpy as np
import math
TRAIN_FILE = './COVID-19-master/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv'
TEST_DATA = []
app = Flask(__name__)

@app.route('/seir/')
def SEIR():
    # the total number of people
    N = 350000
    # per capita daily exposure
    r = 20
    # sensing probability
    beta = 0.8
    # the probability of recovery
    gamma = 0.001
    # incubation
    Te = 14
    # I
    I_0 = 555
    # E
    E_0 = round(I_0 * 0.26)
    # R
    R_0 = 45
    # S
    S_0 = N - I_0 - E_0 - R_0
    # period
    T = 62

    # INI
    INI = (S_0, E_0, I_0, R_0)

    def funcSEIR(inivalue, _):
        Y = np.zeros(4)
        X = inivalue
        # S
        Y[0] = - (r * beta * X[0] * X[2]) / N
        # E
        Y[1] = (r * beta * X[0] * X[2]) / N - X[1] / Te
        # I
        Y[2] = X[1] / Te - gamma * X[2]
        # R
        Y[3] = gamma * X[2]
        return Y

    T_range = np.arange(0, T + 1)

    RES = spi.odeint(funcSEIR, INI, T_range)

    data = {}
    data['x'] = range(T)
    data['y1'] = RES[:, 0]
    data['y2'] = RES[:, 1]
    data['y3'] = RES[:, 2]
    data['y4'] = RES[:, 3]
    with open("./COVID-19-master/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv", 'r') as f:
        reader = list(csv.reader(f))
        length = len(reader) - 1
        content = list(reader[length][4:len(reader[length])])
        data['y5'] = content

    return render_template('seir.html', **data)


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

# data pretreatment
def data_pretreatment():
    # train
    count = 0
    with open(TRAIN_FILE) as f:
        render = csv.reader(f)
        for row in render:
            if count != 502:
                count += 1
                continue
            for i in range(len(row[4:66])):
                if i > 31:
                    break
                temp = []
                temp_row = row[i+4: i + 35]
                count2 = 0
                for item in temp_row:
                    if is_number(item):
                        temp.append(normalization(float(item), 336004, 0))
                    else:
                        temp.append(0.0)
                    count2 += 1

                row_data = np.array(temp, dtype='float32')
                TEST_DATA.append(row_data[:30])
            count += 1




@app.route('/dnn_infer/', methods=['GET'])
def dnn_infer():
    data_pretreatment()
    place = fluid.CUDAPlace(0)
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    pred_array = []

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model("./model", infer_exe)
        iris_feat = np.array(TEST_DATA).astype("float32")
        assert feed_target_names[0] == 'input'
        results = infer_exe.run(inference_program,
                                feed={feed_target_names[0]: np.array(iris_feat)},
                                fetch_list=fetch_targets)

        print("Iris results: (Iris Type)")
        for idx, val in enumerate(results[0]):
            pred = round(anti_normalization(val[0], 336004, 0))
            # pred = val[0]
            pred_array.append(pred)
            # print("%d: %f" % (idx, pred))
        temp_list = list(TEST_DATA[0])
        for i in range(len(temp_list)):
            temp_list[i] = round(anti_normalization(temp_list[i], 336004, 0))
        temp_list.extend(pred_array)
        pred_array = temp_list
    data = {}
    data['infer'] = pred_array
    data['x'] = range(len(pred_array))
    with open("./COVID-19-master/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv", 'r') as f:
        reader = list(csv.reader(f))
        length = len(reader) - 1
        content = list(reader[length][4:len(reader[length])])
        data['real'] = content
        print(len(data['real']))
    f.close()
    return render_template('dnn.html', **data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8383, debug=True)
