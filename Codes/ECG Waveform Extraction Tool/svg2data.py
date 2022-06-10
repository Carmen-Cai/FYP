###This code is for converting SVG to data(waveform)
## Module import (for parsing and multiprocessing) from multiprocessing import Process
from multiprocessing import Pool, cpu_count
from svgpathtools import svg2paths2
from svgpathtools import wsvg
from os import listdir
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy as sp
import pandas as pd
import numpy as np
import cmath
import os
import sys
import re
import hashlib
import time
### Define functions for converting
def pol2cart(polar):
    rho = polar[0]
    phi = polar[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return(x, y)

def pars_info(file):
    print('FILE:',file)
    paths, attributes, svg_attributes = svg2paths2(file)
    pth_tmp = paths[410:]
    # print ('pth_tmp:',pth_tmp)
    return pth_tmp

def chang_corrd(pth_tmp):
    print('chang_corrd')
    lst_tmp = []
    n= 0
    for i in pth_tmp:
        # print('i:',i)
        path_tmp = [pth_tmp[n][0::2]]
        lst_tmp = lst_tmp + path_tmp #get every point data
        n = n + 1
        path_tmp = []
    # print('lst_tmp:',lst_tmp)
    return lst_tmp

def corrd2data(lst_tmp):
    print('corrd2data')
    x_list = []
    y_list = []
    n = 0
    for i in lst_tmp:# a series of [Line(),Line()...]
        # print('I:',i)

        x = []
        y = []
        for i[n] in i:#Line(start=(750+10488j), end=(760+10488j))
            # print('i:',i[n])

            # [0] is first part!
            real_num_s = i[n][0].real
            imag_num_s = i[n][0].imag

            # [1] is end part!
            real_num_e = i[n][1].real
            imag_num_e = i[n][1].imag

            # convert to complex number!
            start_cn = complex(real_num_s, imag_num_s)
            end_cn = complex(real_num_e, imag_num_e)

            # convert to polar coordinates!
            start_pol = cmath.polar(start_cn)
            end_pol = cmath.polar(end_cn)

            # convert to cartesian coordinate!
            start_poi = pol2cart(start_pol)
            end_poi = pol2cart(end_pol)
            # print('Start_poi:',start_poi)
            start_poi_x = [start_poi[0]]
            end_poi_x = [end_poi[0]]

            # print('Start_poi_x:',start_poi_x)
            # print('End_pint_x:',end_poi_x)

            start_poi_y = [start_poi[1]]
            end_poi_y = [end_poi[1]]

            # collect!
            x = x + start_poi_x + end_poi_x
            start_poi_x = []
            end_poi_x = []

            y = y + start_poi_y + end_poi_y
            start_poi_y = []
            end_poi_y = []


        x_list.append(x[1:]) #elimiate the first one
        y_list.append(y[1:])
    print('len:', len(x_list), len(y_list))
        # print('x_list:',x_list)
        # print('y_list:',y_list)
    return x_list, y_list
####Convert file name for DB
# 1)Parsing patient ID from name of pdf file and anonymous!
def gt_id(name):
    p_id = re.findall('id=:\d+',name)
    p_id = ''.join(p_id).replace("=","")
    print('ID:',p_id)
    hexSHA256 = hashlib.sha256(str(p_id).encode('utf-8')).hexdigest()
    # hashSHA.update(p_id)
    # hexSHA256 = hashSHA.hexdigest()
    p_id= hexSHA256[0:20]
    print('ID:', p_id)
    return p_id

# 2)Parsing date from file name
def gt_date(name):
    date = re.findall('date=\d+-\d+-\d+',name)
    date = ''.join(date).replace('Date=','').split("-")
    date[0],date[1],date[2] = date[2],date[1],date[0]
    date = ''.join(date)
    print('date:',date)
    return date
# 3)Parsing time and merge with date for filename
def gt_time(name):
    time = re.findall('Time=\d+%\d+\w+%\d+\w\d+',name)
    time = ''.join(time)
    time = time.replace('Time=','').replace('%3a','')

    return time
# 4)Parsing end number from file name
def gt_endnum(name):
    endnum = re.findall('num\d+',name)
    endnum = ''.join(endnum)

    return endnum

### change Hz of ECG data!(500Hz/s)
def inter(x_list,y_list):
    #ytmp = [float(i) for i in y_list]
    #xtmp = [float(i) for i in x_list]
    y_array = np.array(y_list)
    x_array = np.array(x_list)
    new_length = 1500
    new_x = np.linspace(x_array.min(), x_array.max(), new_length)
    new_y = sp.interpolate.interp1d(x_array, y_array)(new_x)
    x_list = new_x.tolist()
    y_list = new_y.tolist()

    return x_list, y_list

def inter2(x_list,y_list):
    #ytmp = [float(i) for i in y_list]
    #xtmp = [float(i) for i in x_list]
    y_array = np.array(y_list)
    x_array = np.array(x_list)
    new_length = 5000
    new_x = np.linspace(x_array.min(), x_array.max(),new_length)
    new_y = sp.interpolate.interp1d(x_array,y_array)(new_x)
    x_list = new_x.tolist()
    y_list = new_y.tolist()

    return x_list, y_list


def get_point(x_axis,y_axis):
    print("lead I :")
    print(" fisrt point: ", x_axis[3][0],y_axis[3][0])

### fixing data to fit baseline of ECG!
def fix_data(x_list, y_list):
    print('fix_data')
    ## calculate final y value of baseline! checked
    f_bl = y_list[0][-1]
    s_bl = y_list[1][-1]
    t_bl = y_list[2][-1]
    fr_bl = y_list[15][-1]
    ## first, fix y value of first row! checked
    dis_fn3 = f_bl - y_list[3][0]
    if dis_fn3 > 0:
        y_list[3] = list(map(lambda y: y + dis_fn3, y_list[3]))
        x_list[3], y_list[3] = inter(x_list[3],y_list[3])
        y_list[3] = list(map(lambda y: y - y_list[3][0], y_list[3]))
    else:
        y_list[3] = list(map(lambda y: y - dis_fn3, y_list[3]))
        x_list[3], y_list[3] = inter(x_list[3],y_list[3])
        y_list[3] = list(map(lambda y: y - y_list[3][0], y_list[3]))
    dis_fn6 = f_bl - y_list[6][0]
    if dis_fn6 > 0:
        y_list[6] = list(map(lambda y: y + dis_fn6, y_list[6]))
        x_list[6], y_list[6] = inter(x_list[6],y_list[6])
        y_list[6] = list(map(lambda y: y - y_list[6][0], y_list[6]))
    else:
        y_list[6] = list(map(lambda y: y - dis_fn6, y_list[6]))
        x_list[6], y_list[6] = inter(x_list[6], y_list[6])
        y_list[6] = list(map(lambda y: y - y_list[6][0], y_list[6]))
    dis_fn9 = f_bl - y_list[9][0]
    if dis_fn9 > 0:
        y_list[9] = list(map(lambda y: y + dis_fn9, y_list[9]))
        x_list[9], y_list[9] = inter(x_list[9], y_list[9])
        y_list[9] = list(map(lambda y: y - y_list[9][0], y_list[9]))
    else:
        y_list[9] = list(map(lambda y: y - dis_fn9, y_list[9]))
        x_list[9], y_list[9] = inter(x_list[9], y_list[9])
        y_list[9] = list(map(lambda y: y - y_list[9][0], y_list[9]))
    dis_fn12 = f_bl - y_list[12][0]
    if dis_fn12 > 0:
        y_list[12] = list(map(lambda y: y + dis_fn12, y_list[12]))
        x_list[12], y_list[12] = inter(x_list[12], y_list[12])
        y_list[12] = list(map(lambda y: y - y_list[12][0], y_list[12]))
    else:
        y_list[12] = list(map(lambda y: y - dis_fn12, y_list[12]))
        x_list[12], y_list[12] = inter(x_list[12], y_list[12])
        y_list[12] = list(map(lambda y: y - y_list[12][0], y_list[12]))

    ## second, fix y value of second row!
    dis_fn4 = s_bl - y_list[4][0]
    if dis_fn4 > 0:
        y_list[4] = list(map(lambda y: y + dis_fn4, y_list[4]))
        x_list[4], y_list[4] = inter(x_list[4], y_list[4])
        y_list[4] = list(map(lambda y: y - y_list[4][0], y_list[4]))
    else:
        y_list[4] = list(map(lambda y: y - dis_fn4, y_list[4]))
        x_list[4], y_list[4] = inter(x_list[4], y_list[4])
        y_list[4] = list(map(lambda y: y - y_list[4][0], y_list[4]))

    dis_fn7 = s_bl - y_list[7][0]
    if dis_fn7 > 0:
        y_list[7] = list(map(lambda y: y + dis_fn7, y_list[7]))
        x_list[7], y_list[7] = inter(x_list[7], y_list[7])
        y_list[7] = list(map(lambda y: y - y_list[7][0], y_list[7]))
    else:
        y_list[7] = list(map(lambda y: y - dis_fn7, y_list[7]))
        x_list[7], y_list[7] = inter(x_list[7], y_list[7])
        y_list[7] = list(map(lambda y: y - y_list[7][0], y_list[7]))

    dis_fn10 = s_bl - y_list[10][0]
    if dis_fn10 > 0:
        y_list[10] = list(map(lambda y: y + dis_fn10, y_list[10]))
        x_list[10], y_list[10] = inter(x_list[10], y_list[10])
        y_list[10] = list(map(lambda y: y - y_list[10][0], y_list[10]))
    else:
        y_list[10] = list(map(lambda y: y - dis_fn10, y_list[10]))
        x_list[10], y_list[10] = inter(x_list[10], y_list[10])
        y_list[10] = list(map(lambda y: y - y_list[10][0], y_list[10]))

    dis_fn13 = s_bl - y_list[13][0]
    if dis_fn13 > 0:
        y_list[13] = list(map(lambda y: y + dis_fn13, y_list[13]))
        x_list[13], y_list[13] = inter(x_list[13], y_list[13])
        y_list[13] = list(map(lambda y: y - y_list[13][0], y_list[13]))
    else:
        y_list[13] = list(map(lambda y: y - dis_fn13, y_list[13]))
        x_list[13], y_list[13] = inter(x_list[13], y_list[13])
        y_list[13] = list(map(lambda y: y - y_list[13][0], y_list[13]))

    ## thrid, fix y value of third row!checked
    dis_fn5 = t_bl - y_list[5][0]
    if dis_fn5 > 0:
        y_list[5] = list(map(lambda y: y + dis_fn5, y_list[5]))
        x_list[5], y_list[5] = inter(x_list[5], y_list[5])
        y_list[5] = list(map(lambda y: y - y_list[5][0], y_list[5]))
    else:
        y_list[5] = list(map(lambda y: y - dis_fn5, y_list[5]))
        x_list[5], y_list[5] = inter(x_list[5], y_list[5])
        y_list[5] = list(map(lambda y: y - y_list[5][0], y_list[5]))
    dis_fn8 = t_bl - y_list[8][0]
    if dis_fn8 > 0:
        y_list[8] = list(map(lambda y: y + dis_fn8, y_list[8]))
        x_list[8], y_list[8] = inter(x_list[8], y_list[8])
        y_list[8] = list(map(lambda y: y - y_list[8][0], y_list[8]))
    else:
        y_list[8] = list(map(lambda y: y - dis_fn8, y_list[8]))
        x_list[8], y_list[8] = inter(x_list[8], y_list[8])
        y_list[8] = list(map(lambda y: y - y_list[8][0], y_list[8]))

    dis_fn11 = t_bl - y_list[11][0]
    if dis_fn11 > 0:
        y_list[11] = list(map(lambda y: y + dis_fn11, y_list[11]))
        x_list[11], y_list[11] = inter(x_list[11], y_list[11])
        y_list[11] = list(map(lambda y: y - y_list[11][0], y_list[11]))
    else:
        y_list[11] = list(map(lambda y: y - dis_fn11, y_list[11]))
        x_list[11], y_list[11] = inter(x_list[11], y_list[11])
        y_list[11] = list(map(lambda y: y - y_list[11][0], y_list[11]))
    dis_fn14 = t_bl - y_list[14][0]
    if dis_fn14 > 0:
        y_list[14] = list(map(lambda y: y + dis_fn14, y_list[14]))
        x_list[14], y_list[14] = inter(x_list[14], y_list[14])
        y_list[14] = list(map(lambda y: y - y_list[14][0], y_list[14]))
    else:
        y_list[14] = list(map(lambda y: y - dis_fn14, y_list[14]))
        x_list[14], y_list[14] = inter(x_list[14], y_list[14])
        y_list[14] = list(map(lambda y: y - y_list[14][0], y_list[14]))
    ## fourth, fix y value of fourth row!
    dis_fn15 = fr_bl - y_list[16][0]
    if dis_fn15 > 0:
        y_list[16] = list(map(lambda y: y + dis_fn15, y_list[16]))
        x_list[16], y_list[16] = inter2(x_list[16], y_list[16])
        y_list[16] = list(map(lambda y: y - y_list[16][0], y_list[16]))
    else:
        y_list[16] = list(map(lambda y: y - dis_fn15, y_list[16]))
        x_list[16], y_list[16] = inter2(x_list[16], y_list[16])
        y_list[16] = list(map(lambda y: y - y_list[16][0], y_list[16]))

    print('fix_data complete')
    return x_list, y_list

#Adjust scale of ECG data!
def adj(x_list,y_list):
    print('adj')
    candid_1 = y_list[0]
    max_value_1 = max(candid_1)
    min_value_1 = min(candid_1)
    base_1 = abs(max_value_1 - min_value_1)

    y_list[3] = [x/base_1 for x in y_list[3]]
    y_list[6] = [x/base_1 for x in y_list[6]]
    y_list[9] = [x/base_1 for x in y_list[9]]
    y_list[12] = [x/base_1 for x in y_list[12]]

    candid_2 = y_list[1]
    max_value_2 = max(candid_2)
    min_value_2 = min(candid_2)
    base_2 = abs(max_value_2 - min_value_2)

    y_list[4] = [x/base_2 for x in y_list[4]]
    y_list[7] = [x/base_2 for x in y_list[7]]
    y_list[10] = [x/base_2 for x in y_list[10]]
    y_list[13] = [x/base_2 for x in y_list[13]]
    candid_3 = y_list[2]
    max_value_3 = max(candid_3)
    min_value_3 = min(candid_3)
    base_3 = abs(max_value_3 - min_value_3)

    y_list[5] = [x / base_3 for x in y_list[5]]
    y_list[8] = [x / base_3 for x in y_list[8]]
    y_list[11] = [x / base_3 for x in y_list[11]]
    y_list[14] = [x / base_3 for x in y_list[14]]

    candid_4 = y_list[16]
    max_value_4 = max(candid_4)
    min_value_4 = min(candid_4)
    base_4 = abs(max_value_4 - min_value_4)

    y_list[16] = [x / base_4 for x in y_list[16]]

    print('adj complete')
    return x_list, y_list

#make each dataframe for each part of ECG!
def mk_pECG(x_list,y_list):
    p_df_3 = pd.DataFrame(y_list[3])
    p_df_4 = pd.DataFrame(y_list[4])
    p_df_5 = pd.DataFrame(y_list[5])
    p_df_6 = pd.DataFrame(y_list[6])
    p_df_7 = pd.DataFrame(y_list[7])
    p_df_8 = pd.DataFrame(y_list[8])
    p_df_9 = pd.DataFrame(y_list[9])
    p_df_10 = pd.DataFrame(y_list[10])
    p_df_11 = pd.DataFrame(y_list[11])
    p_df_12 = pd.DataFrame(y_list[12])
    p_df_13 = pd.DataFrame(y_list[13])
    p_df_14 = pd.DataFrame(y_list[14])
    p_df_16 = pd.DataFrame(y_list[16])

    return p_df_3, p_df_4, p_df_5, p_df_6, p_df_7, p_df_8, p_df_9, p_df_10, p_df_11, p_df_12, p_df_13, p_df_14, p_df_16

def execute(file):
    try:
        print('TRY')
        #first, change file to data list of ECG!
        pth_tmp = pars_info(file)
        lst_tmp = chang_corrd(pth_tmp)
        x_list, y_list = corrd2data(lst_tmp) #checked
        get_point(x_list,y_list)
        x_list, y_list = fix_data(x_list, y_list)
        x_list, y_list = adj(x_list, y_list)

        # ###third, make subfolder for each patient!
        os.getcwd()
        try:
            # 1)subfolder each patient_id!
            name=file[0:-4]
            os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/csv")
            os.mkdir(name)

        except OSError:
            os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/csv/"  + name)
            p_df_3, p_df_4, p_df_5, p_df_6, p_df_7, p_df_8, p_df_9, \
            p_df_10, p_df_11, p_df_12, p_df_13, p_df_14, p_df_16 = mk_pECG(x_list, y_list)

            p_df_3.to_csv(name+ "_" + "ECGI" + ".gz", index=False,
                          header=False,
                          compression='gzip')
            # print(p_df_3)
            p_df_4.to_csv(name + "_" + "ECGII" + ".gz", index=False,
                          header=False,
                          compression='gzip')
            p_df_5.to_csv(name + "_" + "ECGIII" + ".gz", index=False,
                          header=False,
                          compression='gzip')
            p_df_6.to_csv(name + "_" + "ECGaVR" + ".gz", index=False,
                          header=False,
                          compression='gzip')
            p_df_7.to_csv(name + "_" + "ECGaVL" + ".gz", index=False,
                          header=False,
                          compression='gzip')
            p_df_8.to_csv(name + "_" + "ECGaVF" + ".gz", index=False,
                          header=False,
                          compression='gzip')
            p_df_9.to_csv(name + "_" + "ECGV1" + ".gz", index=False,
                          header=False,
                          compression='gzip')
            p_df_10.to_csv(name + "_" + "ECGV2" + ".gz", index=False,
                           header=False,
                           compression='gzip')
            p_df_11.to_csv(name + "_" + "ECGV3" + ".gz", index=False,
                           header=False,
                           compression='gzip')
            p_df_12.to_csv(name + "_" + "ECGV4" + ".gz", index=False,
                           header=False,
                           compression='gzip')
            p_df_13.to_csv(name + "_" + "ECGV5" + ".gz", index=False,
                           header=False,
                           compression='gzip')
            p_df_14.to_csv(name + "_" + "ECGV6" + ".gz", index=False,
                           header=False,
                           compression='gzip')
            p_df_16.to_csv(name + "_" + "ECGIV" + ".gz", index=False,
                           header=False,
                           compression='gzip')

    # exception handling
    except IndexError:
        print('fail:',file)


if __name__ == '__main__':
    print("Start converting ECG data!")

    os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/")
    search_directory = "svg"
    filelist = listdir(search_directory)

    filelist=[x for x in filelist if '.DS_Store' not in x]
    print('FILELIST:',filelist)

    os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/svg")
    print("Number_of_svg_file :",len(filelist))

    for file in filelist:
        os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/svg")
        execute(file)

