import sys
import pandas as pd
import numpy as np
import librosa, sklearn
import matplotlib.pyplot as plt
import librosa.display
import IPython.display
from datetime import datetime
%matplotlib inline
import cv2
import os
from tqdm import tqdm
import seaborn as sns
from streamlit import caching
plt.rcParams['figure.figsize'] = [16, 12]

#add white to colorlist for drums
colorList = sns.color_palette('hsv', 12)
colorList+=[(1,1,1)]
#balance between low number for precision, high number for stability of pitch/amp calc
hop_length = 2205
#load all files, 
media_name = 's3:/mmacellaiomusic/raw_music/aintnosunshine.mp3'
filename = media_name.split('.')[0].split('/')[1]
print('loading')
channels = {}

# #generate separated files with demucs, move to their own folders
# if not(os.path.isdir(f"demucs/separated/demucs/{filename}")):
#     os.system(f'cd ../../ python -m demucs.separate -d cpu --dl {media_name}')

# if not(os.path.isdir(f"demucs/separated/demucs/{filename}/vocals")):
#     for file in os.listdir(f'demucs/separated/demucs/{filename}'):
#         os.mkdir(f"demucs/separated/demucs/{filename}/{file.split('.')[0]}")
#         os.system(f"mv demucs/separated/demucs/{filename}/{file} demucs/separated/demucs/{filename}/{file.split('.')[0]}")

#load
for i, source in enumerate(['vocals','drums','other','bass']):
    print(source)
    channels['filename'] = filename.split('.')[0]
    channels[source] = {}
#     channels[source]['combined'] = {}
    channels[source][0] = {}
    channels[source][1] = {}
    if 'left.wav' not in os.listdir(f"demucs/separated/demucs/{filename}/{source}"):
#     split to left/right (how to automate if there is no stereo?)
        os.system(f"ffmpeg -i {source}/{source}.wav -map_channel 0.0.0 {source}/left.wav -map_channel 0.0.1 {source}/right.wav")

#     channels[source]['combined']['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{source}/{source}.wav", sr=None)
    channels[source][0]['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{source}/left.wav", sr=None)
    channels[source][1]['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{source}/right.wav", sr=None)
#     channels[source]['stft'] = librosa.stft(channels[source]['audio'])

# generate pitch and amplitude
print('pitch and amplitude')
for source in channels.keys():
    if source != 'filename':
        for channel in [0,1]:
            if source != 'drums':
                #pitch is defaulting to F across the board on Rome. is that the case for OW too?
                chromagram = librosa.feature.chroma_stft(channels[source][channel]['audio'], sr=sr, hop_length=hop_length)
                channels[source][channel]['pitch'] = chromagram
            else:
                channels[source][channel]['pitch'] = np.repeat(12, int(np.floor(len(channels[source][channel]['audio'])/hop_length)))
                
            channels[source][channel]['amp'] = np.array([np.mean(abs(channels[source][channel]['audio'][(s*hop_length):(s*hop_length+hop_length)])) 
                                            for s in range(int(np.floor(len(channels[source][channel]['audio'])/hop_length))-1)])
        
        
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def plotimages(s, dataDict, colorList):
    # take maximum-amplitude pitch for the sample, normalize sample ampl by maximum ampl (or by "other"?)
    # for now, use pitch from channel with larger amp. 
    # how to handle two-channel bass better? revert back to circles?

    import matplotlib.patches as mpatches

    channel = np.argmax([abs(dataDict['bass'][0]['amp'][s]),abs(dataDict['bass'][1]['amp'][s])])
    bgcolor = [c*0.5*(abs(dataDict['bass'][channel]['amp'][s])/
                      abs(max(dataDict['bass'][channel]['amp']))) for c in 
               colorList[np.argmax(dataDict['bass'][channel]['pitch'][:,s])]] 

    fig, ax = plt.subplots(1,1)
                                   
    # add a rectangle
    rect = mpatches.Rectangle([0,0], 3, 3, ec="none", facecolor = bgcolor)
    ax.add_patch(rect)
    
    for channel in [0,1]:
        #drums
        plt.plot(1+channel,1.5, 'o',
                 markerfacecolor = colorList[dataDict['drums'][channel]['pitch'][s]], 
                 markeredgecolor = colorList[dataDict['drums'][channel]['pitch'][s]], 
                 markersize = dataDict['drums'][channel]['amp'][s]*100)

        #grab pitches in reverse order of strength, only using top few
        topchr = np.argsort(dataDict['vocals'][channel]['pitch'][:,s])
        for pitch_rank in range(-1, 1):
            plt.plot(.5+channel*2,0.75, 'o',
                     markerfacecolor = colorList[topchr[pitch_rank]], 
                     markeredgecolor = colorList[topchr[pitch_rank]],
                     markersize = 250*dataDict['vocals'][channel]['amp'][s]*
                     sum(dataDict['vocals'][channel]['pitch'][topchr[range(pitch_rank,1)],s]))

        topchr = np.argsort(dataDict['other'][channel]['pitch'][:,s])
        for pitch_rank in range(-1, 1):
            plt.plot(0.5+channel*2,2.25, 'o',
                     markerfacecolor = colorList[topchr[pitch_rank]], 
                     markeredgecolor = colorList[topchr[pitch_rank]],
                     markersize = 250*dataDict['other'][channel]['amp'][s]*sum(dataDict['other'][channel]['pitch'][topchr[range(pitch_rank,1)],s]))
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.axis('off')
    
    # redraw the canvas
    fig.canvas.draw()

#     # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#     # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return img

digits = len(str(len(channels['vocals'][0]['amp'])))

startt = datetime.now()
if f"{filename}_frames" not in os.listdir():
    os.mkdir(f"{filename}_frames")

start_frame = len(sorted(os.listdir(f'{filename}_frames/')))
size = (1152,864)

if f"{filename}.avi" in os.listdir():
    os.remove(f"{filename}.avi")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
os.system(f"rm -f {filename}.avi")
video = cv2.VideoWriter(f"{filename}cv2.avi", 
                        fourcc, 
                        sr/hop_length, 
                        size)

for i in tqdm(range(500,len(channels['vocals'][0]['amp']))):
    image = plotimages(i, dataDict = channels, colorList = colorList)
    video.write(image.astype('uint8'))


video.release()
cv2.destroyAllWindows()
print(datetime.now()-startt)
if f"{filename}_waud.avi" in os.listdir():
    os.remove(f"{filename}_waud.avi")
os.system(f"ffmpeg -i {filename}cv2.avi -i {media_name} -map 0 -map 1:a -c:v copy -shortest {filename}_waud.avi")
