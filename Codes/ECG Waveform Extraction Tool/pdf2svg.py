###This code is for converting PDF file to SVG file.
##import library for converting ##
from multiprocessing import Process
from multiprocessing import Pool, cpu_count
from os import listdir
import time
import os

folder="NoRecur"
## execute function for converting ##
def execute(filelist):

    command = '/Applications/Inkscape.app/Contents/MacOS/inkscape  --export-filename=/Users/Cai/Desktop/FYP/ECGpdf2data/svg/'+folder+"/"  + filelist[0:-4] \
              + ".svg /Users/cai/Desktop/FYP/ECGpdf2data/dataset/PDF_ECG/"+folder+"/" + filelist

    os.system(command)

    # directory is the physical directory where the file is located
if __name__ == '__main__':

    print("Start convert ECG data!")

    # search_directory = "Recur"
    filelist = listdir(os.chdir("/Users/Cai/Desktop/FYP/ECGpdf2data/dataset/PDF_ECG/"+folder))
    filelist=[x for x in filelist if '.DS_Store' not in x]

    processor = cpu_count()
    proc = os.getpid()

    print("proc_id",proc)
    print("Number of processor:",processor)
    print("Number_of_pdf_file :", len(filelist))
    pool = Pool(processes = cpu_count())

    startTime = int(time.time())


    pool.map(execute, filelist)

    print("Pdf2svg finished!")
    endTime = int(time.time())
    print("Total converting time", (endTime - startTime))
