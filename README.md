# FYP
## In general:
1. The ECG Waveform Extraction Tool: to extarct waveforms from ECG PDF Reports 
2. The GAA-CNN model: to generate ECG waveforms and conduct basic diagnosis from ECG Paper Reports 


## Datasets(8:1:1): Saved in Google Drive.
1. ECG PDF Reports (cropped) (975)
2. Extracted ECG PDF Waveforms (975)
3. Digitized ECG Paper Reports for the GAA (975)

5. Digitized ECG Paper Reports for the CNN (172)
6. Diagnosis Label (172)

The datasets have been split in advanced and saved in 'GAA data' and 'CNN data' respectively, regarding the purpose.

### GAA training:
1. Extracted PDF Waveforms: /content/drive/My Drive/FYP_Yixin_Cai/GAA_data/train/extracted_waveforms/extracted
2. Cropped PDF: /content/drive/My Drive/FYP_Yixin_Cai/GAA_data/train/cropped_png/cropped
3. Digitized Paper: /content/drive/My Drive/FYP_Yixin_Cai/GAA_data/train/digitized_paper

### CNN training:
1. Extracted waveforms: /content/drive/My Drive/FYP_Yixin_Cai/CNN_data/train/extracted_digitized_waveform/binary_image_1

N.B. Validation and Test datasets follows the same address rule, with the folder name 'train' changed to 'val' or 'test'.


## Codes: Saved in Github.
1. The ECG Waveform Extraction Tool (PyCharm)
2. The GAA-CNN model (CoLab)


## Checkpoints:
saved in 'GAA data' and 'CNN data' repesctively. 

N.B. The 'GAA_epoch_1.pkl' is an ideal model trained so far for generating waveforms from paper reports.


## Paper works:
All saved in Google Drive.

