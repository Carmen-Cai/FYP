from multiprocessing import Pool, cpu_count
import cv2
import os
from os import listdir
import time


def execute(file):

    if file[-4:0]==".jpg":
        filepath = '/Users/Cai/Desktop/FYP/ECGpdf2data/dataset/ECG_split_train_test/testing/' + file
        savepath = '/Users/Cai/Desktop/FYP/ECGpdf2data/dataset/ECG_split_train_test/testing/' + file[0:-4] + '.png'

        img = cv2.imread(filepath)
        cv2.imwrite(savepath, img)


if __name__ == '__main__':
    # filelist = listdir(os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/dataset/PDF_ECG/NoRecur"))
    filelist = listdir(os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/dataset/ECG_split_train_test/testing"))
    filelist = [x for x in filelist if '.DS_Store' not in x]
    # print(filelist)
    processor = cpu_count()
    pool = Pool(processes=cpu_count())
    startTime = int(time.time())

    pool.map(execute, filelist)

    endTime = int(time.time())


    print("saved")