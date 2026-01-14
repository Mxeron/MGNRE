from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import k_means
import time
import warnings
from scipy.spatial.distance import cdist
warnings.filterwarnings('ignore')


def get_radius(gb):
    real_gb = gb[:,:-1]
    center = real_gb.mean(0)
    radius = max(np.sqrt(np.sum((real_gb - center) ** 2, axis=1)))
    return radius

def spilt_ball(data):
    m = data.shape[1]
    cluster = k_means(X=data[:,:m - 1], init='k-means++', n_clusters=2, n_init=5)[1]
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]

def spilt_ball_2(data):
    real_data = data[:,:-1]
    ball1 = []
    ball2 = []
    D = cdist(real_data,real_data)
    r, c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_density_volume(gb):
    real_gb = gb[:,:-1]
    num = len(real_gb)
    center = real_gb.mean(0)
    sum_radius = np.sum(np.sqrt(np.sum((real_gb - center) ** 2, axis=1)))
    mean_radius = sum_radius / num
    if mean_radius != 0:
        density_volume = num / sum_radius
    else:
        density_volume = num

    return density_volume


def division_ball(gb_list):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) >= 4:
            ball_1, ball_2 = spilt_ball_2(gb)
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb)
                continue
            density_parent = get_density_volume(gb)
            density_child_1 = get_density_volume(ball_1)
            density_child_2 = get_density_volume(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = (w1 * density_child_1 + w2 * density_child_2)
            t2 = (w_child > density_parent)
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)

    return gb_list_new


def normalized_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb in gb_list:
        if len(gb) < 2:
            gb_list_temp.append(gb)
        else:
            ball_1, ball_2 = spilt_ball_2(gb)
            if get_radius(gb) <= 2 * radius_detect:
                gb_list_temp.append(gb)
            else:
                gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp


def get_GB(data):
    data_num = data.shape[0]
    index = np.array(range(data_num)).reshape(data_num, 1)
    data = np.hstack((data, index))
    gb_list_temp = [data]
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_ball(gb_list_temp)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
    radius = []
    for gb in gb_list_temp:
        if len(gb) >= 2:
            radius.append(get_radius(gb))

    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)
    
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball(gb_list_temp, radius_detect)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    gb_list_final = gb_list_temp
    return gb_list_final

if __name__ == '__main__':
    load_data = loadmat(r"D:\outlier_datasets\Numerical\diabetes_tested_positive_26_variant1.mat")  # 加载数据集
    # load_data = loadmat('Lymphography.mat')
    data = load_data['trandata']
    n, m = data.shape
    print('样本个数：', n, '总属性个数', m)
    label = data[:, m - 1]
    print('异常点个数:', label.sum())
    trandata = data[:, :m - 1]
    scaler = MinMaxScaler()
    trandata = scaler.fit_transform(trandata)

    start_time = time.time()
    # 还没有把index带上！
    gb_final = get_GB(trandata)
    end_time = time.time()
    print("运行时间：", end_time - start_time)
    print("粒球个数：", len(gb_final))