from multiprocessing import Pool, cpu_count
from svgpathtools import svg2paths2
from svgpathtools import wsvg
from os import listdir
import scipy.interpolate
import scipy as sp
import numpy as np
import cmath
import os

import time
from PIL import Image,ImageDraw,ImageOps
folder="NoRecur/" #could be your own folder

### Define functions for converting
def pol2cart(polar):
    rho = polar[0]
    phi = polar[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return(x, y)

def pars_info(file):

    filepath="/Users/Cai/Desktop/FYP/ECGpdf2data/svg/NoRecur/"+file[0:-4]+".svg"

    paths, attributes, svg_attributes = svg2paths2(filepath)
    pth_tmp = paths[410:]

    return pth_tmp

def chang_corrd(pth_tmp):  #change it to list

    lst_tmp = []
    n= 0

    for i in pth_tmp:

        path_tmp = [pth_tmp[n][0::2]]

        lst_tmp = lst_tmp + path_tmp #get every point data
        n = n + 1
        path_tmp = []
    return lst_tmp

def corrd2data(lst_tmp): #covert each part to cartesian then store in x[], j[]

    x_list = []
    y_list = []
    n = 0

    for i in lst_tmp:# a series of [Line(),Line()...]

        x = []
        y = []
        for i[n] in i:#Line(start=(26070+2714j), end=(26075+2718j))

            # [0] is start part!
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

            start_poi_x = [start_poi[0]]#[26069.999999999996]
            end_poi_x = [end_poi[0]]#[26075.000000000004]


            start_poi_y = [start_poi[1]]
            end_poi_y = [end_poi[1]]

            # collect!
            x = x + start_poi_x + end_poi_x
            start_poi_x = []
            end_poi_x = []

            y = y + start_poi_y + end_poi_y
            start_poi_y = []
            end_poi_y = []


        x_list.append(x[2:]) #elimiate the first one
        y_list.append(y[2:])

    return x_list, y_list

def inter2(x_list,y_list):
    #ytmp = [float(i) for i in y_list]
    #xtmp = [float(i) for i in x_list]
    y_array = np.array(y_list)
    x_array = np.array(x_list)
    new_length = 1500
    new_x = np.linspace(x_array.min(), x_array.max(),new_length)
    new_y = sp.interpolate.interp1d(x_array,y_array)(new_x)
    x_list = new_x.tolist()
    y_list = new_y.tolist()

    return x_list, y_list

### change Hz of ECG data!(500Hz/s)
def inter(x_list,y_list):
    y_array = np.array(y_list)
    x_array = np.array(x_list)
    new_length = 1236
    new_x = np.linspace(x_array.min(), x_array.max(), new_length)
    new_y = sp.interpolate.interp1d(x_array, y_array)(new_x)
    x_list = new_x.tolist()
    y_list = new_y.tolist()

    return x_list, y_list

### fixing data to fit baseline of ECG! checked
def fix_data(x_list, y_list):

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

    else: #  indeed <0
        y_list[3] = list(map(lambda y: y - dis_fn3, y_list[3]))
        x_list[3], y_list[3] = inter(x_list[3],y_list[3])
        y_list[3] = list(map(lambda y: y - y_list[3][0], y_list[3]))


    dis_fn6 = f_bl - y_list[6][0]
    if dis_fn6 > 0:
        # y_list[6] = list(map(lambda y: y + dis_fn6, y_list[6]))
        x_list[6], y_list[6] = inter(x_list[6],y_list[6])
        y_list[6] = list(map(lambda y: y - y_list[6][0]- dis_fn6, y_list[6]))
    else:
        # print("<0")
        # y_list[6] = list(map(lambda y: y - dis_fn6, y_list[6]))
        x_list[6], y_list[6] = inter(x_list[6], y_list[6])
        y_list[6] = list(map(lambda y: y - y_list[6][0]- dis_fn6, y_list[6]))

    dis_fn9 = f_bl - y_list[9][0]
    if dis_fn9 > 0:
        # y_list[9] = list(map(lambda y: y + dis_fn9, y_list[9]))
        x_list[9], y_list[9] = inter(x_list[9], y_list[9])
        y_list[9] = list(map(lambda y: y - y_list[9][0]+ dis_fn9, y_list[9]))
    else:
        # y_list[9] = list(map(lambda y: y - dis_fn9, y_list[9]))
        x_list[9], y_list[9] = inter(x_list[9], y_list[9])
        y_list[9] = list(map(lambda y: y - y_list[9][0]- dis_fn9, y_list[9]))
    # print('y_list[12][0]:', y_list[12][0])
    dis_fn12 = f_bl - y_list[12][0]
    if dis_fn12 > 0:
        # y_list[12] = list(map(lambda y: y + dis_fn12, y_list[12]))
        x_list[12], y_list[12] = inter(x_list[12], y_list[12])
        y_list[12] = list(map(lambda y: y - y_list[12][0]- dis_fn12, y_list[12]))
    else:
        # y_list[12] = list(map(lambda y: y - dis_fn12, y_list[12]))
        x_list[12], y_list[12] = inter(x_list[12], y_list[12])
        y_list[12] = list(map(lambda y: y - y_list[12][0] + dis_fn12, y_list[12]))

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
    # print('y_list[7][0]:', y_list[7][0])
    if dis_fn7 > 0:
        # y_list[7] = list(map(lambda y: y + dis_fn7, y_list[7]))
        x_list[7], y_list[7] = inter(x_list[7], y_list[7])
        y_list[7] = list(map(lambda y: y - y_list[7][0]+ dis_fn7, y_list[7]))
    else:
        # y_list[7] = list(map(lambda y: y - dis_fn7, y_list[7]))
        x_list[7], y_list[7] = inter(x_list[7], y_list[7])
        y_list[7] = list(map(lambda y: y - y_list[7][0]- dis_fn7, y_list[7]))

    # print('y_list[10][0]:', y_list[10][0])
    dis_fn10 = s_bl - y_list[10][0]
    if dis_fn10 > 0:
        # y_list[10] = list(map(lambda y: y + dis_fn10, y_list[10]))
        x_list[10], y_list[10] = inter(x_list[10], y_list[10])
        y_list[10] = list(map(lambda y: y - y_list[10][0]+ dis_fn10, y_list[10]))
    else:
        # y_list[10] = list(map(lambda y: y - dis_fn10, y_list[10]))
        x_list[10], y_list[10] = inter(x_list[10], y_list[10])
        y_list[10] = list(map(lambda y: y - y_list[10][0]- dis_fn10, y_list[10]))
    dis_fn13 = s_bl - y_list[13][0]
    if dis_fn13 > 0:
        # y_list[13] = list(map(lambda y: y + dis_fn13, y_list[13]))
        x_list[13], y_list[13] = inter(x_list[13], y_list[13])
        y_list[13] = list(map(lambda y: y - y_list[13][0]+ dis_fn13, y_list[13]))
    else:
        # y_list[13] = list(map(lambda y: y - dis_fn13, y_list[13]))
        x_list[13], y_list[13] = inter(x_list[13], y_list[13])
        y_list[13] = list(map(lambda y: y - y_list[13][0]- dis_fn13, y_list[13]))

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
        # y_list[8] = list(map(lambda y: y + dis_fn8, y_list[8]))
        x_list[8], y_list[8] = inter(x_list[8], y_list[8])
        y_list[8] = list(map(lambda y: y - y_list[8][0]+ dis_fn8, y_list[8]))
    else:
        # y_list[8] = list(map(lambda y: y - dis_fn8, y_list[8]))
        x_list[8], y_list[8] = inter(x_list[8], y_list[8])
        y_list[8] = list(map(lambda y: y - y_list[8][0]- dis_fn8, y_list[8]))

    dis_fn11 = t_bl - y_list[11][0]
    if dis_fn11 > 0:
        # y_list[11] = list(map(lambda y: y + dis_fn11, y_list[11]))
        x_list[11], y_list[11] = inter(x_list[11], y_list[11])
        y_list[11] = list(map(lambda y: y - y_list[11][0]+ dis_fn11, y_list[11]))
    else:
        # y_list[11] = list(map(lambda y: y - dis_fn11, y_list[11]))
        x_list[11], y_list[11] = inter(x_list[11], y_list[11])
        y_list[11] = list(map(lambda y: y - y_list[11][0]- dis_fn11, y_list[11]))


    dis_fn14 = t_bl - y_list[14][0]
    if dis_fn14 > 0:

        x_list[14], y_list[14] = inter(x_list[14], y_list[14])
        y_list[14] = list(map(lambda y: y - y_list[14][0]+ dis_fn14, y_list[14]))
    else:
        # y_list[14] = list(map(lambda y: y - dis_fn14, y_list[14]))
        x_list[14], y_list[14] = inter(x_list[14], y_list[14])
        y_list[14] = list(map(lambda y: y - y_list[14][0]- dis_fn14, y_list[14]))

    ## fourth, fix y value of fourth row! checked

    dis_fn15 = fr_bl - y_list[16][0]
    if dis_fn15 > 0:
        y_list[16] = list(map(lambda y: y + dis_fn15, y_list[16]))
        x_list[16], y_list[16] = inter2(x_list[16], y_list[16])
        y_list[16] = list(map(lambda y: y - y_list[16][0], y_list[16]))
    else:
        y_list[16] = list(map(lambda y: y - dis_fn15, y_list[16]))
        x_list[16], y_list[16] = inter2(x_list[16], y_list[16])
        y_list[16] = list(map(lambda y: y - y_list[16][0], y_list[16]))


    return x_list, y_list

def crop_png(x_list,y_list,file):

    # cropped pdf size： 1236* 0.2*[max()-min()+600]

    im = Image.open("/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/origin_png/NoRecur/" + file)
    # im = Image.open('/Users/Cai/Desktop/ECG1.png')
    img_width, img_height = im.size


    upper_left_x, upper_left_y = img_width * (x_list[0] / 27930), img_height * ((21590 - 300 -max(y_list)) / 21590)

    lower_right_x, lower_right_y= img_width * ( x_list[-1] / 27930), img_height * ((21590 + 300 - min(y_list)) / 21590)


    im_crop = im.crop((upper_left_x, upper_left_y, lower_right_x, lower_right_y))
    img_width, img_height = im_crop.size

    basewidth = 1236
    wpercent = (basewidth / float(im_crop.size[0]))

    hsize = int((float(im_crop.size[1]) * float(wpercent)))

    img = im_crop.resize((basewidth, hsize), Image.ANTIALIAS)
    print("img size: ",img.size)

    return img,img.size[1]

def plot_svg2png(height,y_list,file):
    print('start extracting waveform')

    i = 0
    x_axis = []
    y_axis = []
    y_min= int(0-min(y_list)) if min(y_list)<=0 else int(min(y_list))

    base=int((y_min+300)*0.2)

    for x_pos, y in enumerate(y_list):

        y_pos = 0.2 * (float(y))
        x_axis.append(x_pos)
        y_axis.append(y_pos)

    img = Image.new('RGB', (1236, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    for x_pos in range(0, 1236, 1):
        y_pos=round(y_axis[x_pos])
        last_y_pos=round(y_axis[x_pos-1])
        if x_pos>=2:
            if (y_pos-last_y_pos)>1 :
                draw.line((x_pos-1,round(y_axis[x_pos-1])+base,x_pos,
                           round(y_axis[x_pos])+base),fill=(0,0,0),joint="curve")
            if (y_pos - last_y_pos) < -1:
                draw.line((x_pos - 1, round(y_axis[x_pos - 1]) + base, x_pos,
                           round(y_axis[x_pos]) + base),fill=(0,0,0), joint="curve")
        img.putpixel((x_pos, round(y_axis[x_pos])+base), (0,0,0))
    new_img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return new_img


def execute(file):
    try:
        print('TRY '+file) #file="xxx.png"
        # first, change file to data list of ECG!
        pth_tmp = pars_info(file)
        lst_tmp = chang_corrd(pth_tmp)
        x_list_origin, y_list_origin = corrd2data(lst_tmp) #checked

        x_list_crop = x_list_origin.copy()
        y_list_crop = y_list_origin.copy()
        x_list_final, y_list_final = fix_data(x_list_origin, y_list_origin)

        # # for Lead I
        cropped_png,height = crop_png(x_list_crop[3], y_list_crop[3], file)
        waveform = plot_svg2png(height,y_list_final[3], file)
        waveform.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_I.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] + "_I.png")
        # # # for Lead aVR
        cropped_png, height = crop_png(x_list_crop[6], y_list_crop[6], file)
        waveform = plot_svg2png(height, y_list_final[6], file)
        waveform.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_aVR.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] + "_aVR.png")
        # #
        # # # for Lead V1
        cropped_png, height = crop_png(x_list_crop[9], y_list_crop[9], file)
        waveform = plot_svg2png(height, y_list_final[9], file)
        waveform.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_V1.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_V1.png")
        # # for Lead V4
        cropped_png, height = crop_png(x_list_crop[12], y_list_crop[12], file)
        waveform = plot_svg2png(height, y_list_final[12], file)
        waveform.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_V4.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_V4.png")

        # # # for Lead II
        cropped_png, height = crop_png(x_list_crop[4], y_list_crop[4], file)
        waveform = plot_svg2png(height, y_list_final[4], file)
        waveform.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_II.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_II.png")

        # # # for Lead aVL
        cropped_png, height = crop_png(x_list_crop[7], y_list_crop[7], file)
        waveform = plot_svg2png(height, y_list_final[7], file)
        waveform.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_aVL.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_aVL.png")

        # # # for Lead V2
        cropped_png, height = crop_png(x_list_crop[10], y_list_crop[10], file)
        waveform = plot_svg2png(height, y_list_final[10], file)
        waveform.save(
            '/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_V2.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_V2.png")

        # # # for Lead V5
        cropped_png, height = crop_png(x_list_crop[13], y_list_crop[13], file)
        waveform = plot_svg2png(height, y_list_final[13], file)
        waveform.save(
            '/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_V5.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_V5.png")

        # # # for Lead III
        cropped_png, height = crop_png(x_list_crop[5], y_list_crop[5], file)
        waveform = plot_svg2png(height, y_list_final[5], file)
        waveform.save(
            '/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_III.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_III.png")

        # # # for Lead aVF
        cropped_png, height = crop_png(x_list_crop[8], y_list_crop[8], file)
        waveform = plot_svg2png(height, y_list_final[8], file)
        waveform.save(
            '/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_aVF.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_aVF.png")

        # # # for Lead V3
        cropped_png, height = crop_png(x_list_crop[11], y_list_crop[11], file)
        waveform = plot_svg2png(height, y_list_final[11], file)
        waveform.save(
            '/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_V3.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_V3.png")

        # # # for Lead V6
        cropped_png, height = crop_png(x_list_crop[14], y_list_crop[14], file)
        waveform = plot_svg2png(height, y_list_final[14], file)
        waveform.save(
            '/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/extracted_waveforms/'+folder + file[0:-4] +'_V6.png')
        cropped_png.save('/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/cropped_png/'+folder + file[0:-4] +"_V6.png")

        #
        print("finish")

    # exception handling
    except IndexError:
        print('fail:',file)

if __name__ == '__main__':
    print("Start converting ECG data!")


    filelist_1 = listdir(os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/data/train/origin_png/NoRecur/"))


    filelist_1=[x for x in filelist_1 if '.DS_Store' not in x]



    processor = cpu_count()
    pool = Pool(processes = cpu_count())
    print("Number of processor:",processor)
    startTime = int(time.time())
    pool.map(execute, filelist_1)


    endTime = int(time.time())
    print("Total converting time", (endTime - startTime))
