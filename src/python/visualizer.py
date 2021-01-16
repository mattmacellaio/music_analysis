import sys
import pandas as pd
import numpy as np
import librosa, sklearn
import matplotlib.pyplot as plt
import librosa.display
import IPython.display
from datetime import datetime
import cv2
import os
from tqdm import tqdm
import seaborn as sns
from streamlit import caching

colorList = sns.color_palette('hsv', 12)
colorList+=[(1,1,1)]
hop_length = 128
#load all files, 
filename = 'rome'

print('loading')
channels = {}
for i, channel in enumerate(['vocals','drums','other','bass']):
    channels[channel] = {}
    channels['filename'] = filename
    channels[channel]['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{channel}.wav")
#     channels[channel]['stft'] = librosa.stft(channels[channel]['audio'])

print('pitch/amp analysis')
for channel in channels.keys():
    if channel != 'filename':
        if channel != 'drums':
            chromagram = librosa.feature.chroma_stft(channels[channel]['audio'], sr=sr, hop_length=hop_length)
            topchr = [np.argmax(chromagram[:,s])for s in range(chromagram.shape[1])]
            channels[channel]['pitch'] = topchr
        else:
            channels[channel]['pitch'] = np.repeat(12, int(np.floor(len(channels[channel]['audio'])/hop_length)))
        channels[channel]['amp'] = [np.mean(abs(channels[channel]['audio'][(s*hop_length):(s*hop_length+hop_length)])) for s in range(int(np.floor(len(channels[channel]['audio'])/hop_length))-1)]
    
    
def plotimages(s, dataDict, colorList):
    plt.rcParams['axes.facecolor']=[i*0.2*(abs(dataDict['bass']['amp'][s])/abs(max(dataDict['other']['amp']))) for i in colorList[dataDict['bass']['pitch'][s]]]
    plt.rcParams['savefig.facecolor']=[i*0.2*(abs(dataDict['bass']['amp'][s])/abs(max(dataDict['other']['amp']))) for i in colorList[dataDict['bass']['pitch'][s]]]
    fig, ax = plt.subplots(1,1)

    plt.plot(1,2, 'o',
             markerfacecolor = colorList[dataDict['drums']['pitch'][s]], 
             markeredgecolor = colorList[dataDict['drums']['pitch'][s]], 
             markersize = dataDict['drums']['amp'][s]*100)
    plt.plot(1,1, 'o',
             markerfacecolor = colorList[dataDict['vocals']['pitch'][s]], 
             markeredgecolor = colorList[dataDict['vocals']['pitch'][s]], 
             markersize = dataDict['vocals']['amp'][s]*100)
    plt.plot(2,2, 'o',
             markerfacecolor = colorList[dataDict['other']['pitch'][s]], 
             markeredgecolor = colorList[dataDict['other']['pitch'][s]], 
             markersize = dataDict['other']['amp'][s]*100)
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.axis('off')
    plt.savefig(f"{dataDict['filename']}_frames/{s:04d}.jpg", facecolor=ax.get_facecolor(), edgecolor='none')
    plt.close()

    

startt = datetime.now()
if f"{filename}_tmp.avi" in os.listdir():
    os.remove(f"{filename}_tmp.avi")

print('first round of video frames')
os.mkdir(f"{filename}_frames")
for i in tqdm(range(0, 2000)):
    plotimages(i, dataDict = channels, colorList = colorList)

image = cv2.imread(f'{filename}_frames/0.jpg')
height, width, layers = image.shape
size = (width,height)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter(f"{filename}.avi", 
                        fourcc, 
                        sr/hop_length, 
                        size)
for i in tqdm(range(len(sorted(os.listdir(f'{filename}_frames'))))):
    image = cv2.imread(f'{filename}_frames/{i:04d}.jpg')
    video.write(image.astype('uint8'))

video.release()
cv2.destroyAllWindows()
start_frame = len(sorted(os.listdir(f'{filename}_frames/')))-1

while start_frame < len(channels['vocals']['amp']):
    print(f"{start_frame}")
    for i in tqdm(range(start_frame, min(start_frame+2000, len(channels['vocals']['amp'])))):
        plotimages(i, dataDict = channels, colorList = colorList)


    image = cv2.imread(f'{filename}_frames/0.jpg')
    height, width, layers = image.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(f"{filename}_tmp.avi", 
                            fourcc, 
                            sr/hop_length, 
                            size)
    for i in tqdm(range(len(sorted(os.listdir(f'{filename}_frames'))))):
        image = cv2.imread(f'{filename}_frames/{i:04d}.jpg')
        video.write(image.astype('uint8'))

    video.release()
    cv2.destroyAllWindows()
    caching.clear_cache()
    
    print(datetime.now()-startt)
    clip_1 = VideoFileClip(f"{filename}.avi")
    clip_2 = VideoFileClip(f"{filename}_tmp.avi")
    final_clip = concatenate_videoclips([clip_1,clip_2])
    final_clip.write_videofile(f"{filename}.avi")
    start_frame = len(sorted(os.listdir(f'{filename}_frames/')))-1
