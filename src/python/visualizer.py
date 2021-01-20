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
plt.rcParams['figure.figsize'] = [16, 12]

#add white to colorlist for drums
colorList = sns.color_palette('hsv', 12)
colorList+=[(1,1,1)]
#balance between low number for precision, high number for stability of pitch/amp calc
hop_length = 512
#load all files, 
prefix = '' #or s3:/mmacellaiomusic/
media_name = f'{prefix}raw_music/mfdoom_thatsthat.mp3'
splitStereo = {'drums':False, 'vocals':False, 'other': True}
sampBlend = {'drums':1, 'vocals':4, 'other': 3, 'bass':4}
filename = media_name.split('/')[-1].split('.')[0]
channels = {}


# generate separated files with demucs, move to their own folders
# if not(os.path.isdir(f"demucs/separated/demucs/{filename}")):
#     print('Separating sources')
#     os.system(f'python -m demucs.separate -d cpu --dl raw_music/{media_name}')

if not(os.path.isdir(f"demucs/separated/demucs/{filename}/vocals")):
    for file in os.listdir(f'demucs/separated/demucs/{filename}'):
        os.mkdir(f"demucs/separated/demucs/{filename}/{file.split('.')[0]}")
        os.system(f"mv demucs/separated/demucs/{filename}/{file} demucs/separated/demucs/{filename}/{file.split('.')[0]}")

#load
print('Loading, splitting stereo audio')

for i, source in enumerate(['vocals','drums','other','bass']):
#     print(source)
    channels['filename'] = filename.split('.')[0]
    channels[source] = {}
#     channels[source]['combined'] = {}
    channels[source][0] = {}
    channels[source][1] = {}
    if 'left.wav' not in os.listdir(f"demucs/separated/demucs/{filename}/{source}"):
#     split to left/right (how to automate if there is no stereo?)
        os.system(f"ffmpeg -i demucs/separated/demucs/{filename}/{source}/{source}.wav -map_channel 0.0.0 demucs/separated/demucs/{filename}/{source}/left.wav -map_channel 0.0.1 demucs/separated/demucs/{filename}/{source}/right.wav")

#     channels[source]['combined']['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{source}/{source}.wav", sr=None)
    channels[source][0]['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{source}/left.wav", sr=None)
    channels[source][1]['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{source}/right.wav", sr=None)
#     channels[source]['stft'] = librosa.stft(channels[source]['audio'])

# generate pitch and amplitude
print('Analysis')
for source in channels.keys():
    if source != 'filename':
        for channel in [0,1]:
            if source != 'drums':
                chromagram = librosa.feature.chroma_stft(channels[source][channel]['audio'], sr=sr, hop_length=hop_length)
                channels[source][channel]['pitch'] = chromagram
            else:
                channels[source][channel]['pitch'] = np.repeat(12, int(np.floor(len(channels[source][channel]['audio'])/hop_length)))
                
            channels[source][channel]['amp'] = np.array([np.mean(abs(channels[source][channel]['audio'][(s*hop_length):(s*hop_length+hop_length)])) 
                                            for s in range(int(np.floor(len(channels[source][channel]['audio'])/hop_length))-1)])
        
def plotimages(s, dataDict, colorList, splitStereo, sampBlend = 2, pitchShow = 2):
    # take maximum-amplitude pitch for the sample, normalize sample ampl by maximum ampl (or by "other"?)
    # for now, use pitch from channel with larger amp. 
    # how to handle two-channel bass better? revert back to circles?

    import matplotlib.patches as mpatches

    channel = np.argmax([abs(dataDict['bass'][0]['amp'][s]),abs(dataDict['bass'][1]['amp'][s])])
    
    #blend across multiple windowed samples to reduce jitter and flashing
    samples_blend = list(range(max([s-sampBlend['bass'], 0]),min([s+sampBlend['bass']+1, len(dataDict['bass'][0]['amp'])])))
    
    #background color = mean of bass pitches across blended samples
#     modulated by relative amplitude to maximum bass amplitude
#or normalize to channel[source][channel]['centroid']?
    basscolor = sns.color_palette('husl', 12)
    bgcolor = [c*0.4 for c in basscolor[np.argmax([np.mean(dataDict['bass'][0]['pitch'][v,samples_blend]) 
                                    for v in range(dataDict['bass'][0]['pitch'].shape[0])])]] 

    fig, ax = plt.subplots(1,1)
                                   
    # add a rectangle
    rect = mpatches.Rectangle([-2,-2], 7, 7, edgecolor="none", facecolor = bgcolor)
    ax.add_patch(rect)
    
    if splitStereo['drums']:
        samples_blend = list(range(max([s-sampBlend['drums'], 0]),min([s+sampBlend['drums']+1, len(dataDict['drums'][0]['amp'])])))

        for channel in [0,1]:
            #drums
            plt.plot(1+channel,1.5, 'o',
                     markerfacecolor = colorList[dataDict['drums'][channel]['pitch'][s]], 
                     markeredgecolor = colorList[dataDict['drums'][channel]['pitch'][s]], 
                     markersize = np.mean(dataDict['drums'][channel]['amp'][samples_blend])*250)
    else:
        samples_blend = list(range(max([s-sampBlend['drums'], 0]),min([s+sampBlend['drums']+1, len(dataDict['drums'][0]['amp'])])))
        plt.plot(1.5,1.5, 'o',
                     markerfacecolor = colorList[dataDict['drums'][0]['pitch'][s]], 
                     markeredgecolor = colorList[dataDict['drums'][0]['pitch'][s]], 
                     markersize = np.mean(dataDict['drums'][0]['amp'][samples_blend])*250)

    #
    if splitStereo['vocals']:
        samples_blend = list(range(max([s-sampBlend['vocals'], 0]),min([s+sampBlend['vocals']+1, len(dataDict['vocals'][0]['amp'])])))

        for channel in [0,1]:
    #grab pitches in reverse order of strength, only using top few
            pitches_blended = np.mean(dataDict['vocals'][channel]['pitch'][:, samples_blend], axis = 1)
            pitch_inds = np.argsort(pitches_blended)
            for pitch_rank in range(-pitchShow, 0):
                plt.plot(.5+channel*2, #to handle l/r
                         0.75, #height
                         'o',
                         markerfacecolor = colorList[pitch_inds[pitch_rank]], 
                         markeredgecolor = colorList[pitch_inds[pitch_rank]],
                         markersize = 400*np.mean(dataDict['vocals'][channel]['amp'][samples_blend])*
                         sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))
    else:
        pitches_blended = np.mean(dataDict['vocals'][0]['pitch'][:, samples_blend], axis = 1)
        pitch_inds = np.argsort(pitches_blended)
        for pitch_rank in range(-pitchShow, 0):
            plt.plot(1.5, #to handle l/r
                     0.75, #height
                     'o',
                     markerfacecolor = colorList[pitch_inds[pitch_rank]], 
                     markeredgecolor = colorList[pitch_inds[pitch_rank]],
                     markersize = 400*np.mean(dataDict['vocals'][0]['amp'][samples_blend])*
                     sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))

    if splitStereo['other']:
        samples_blend = list(range(max([s-sampBlend['other'], 0]),min([s+sampBlend['other']+1, len(dataDict['other'][0]['amp'])])))

        for channel in [0,1]:
    #grab pitches in reverse order of strength, only using top few
            pitches_blended = np.mean(dataDict['other'][channel]['pitch'][:, samples_blend], axis = 1)
            pitch_inds = np.argsort(pitches_blended)
            for pitch_rank in range(-pitchShow, 0):
                plt.plot(.5+channel*2, #to handle l/r
                         2.25, #height
                         'o',
                         markerfacecolor = colorList[pitch_inds[pitch_rank]], 
                         markeredgecolor = colorList[pitch_inds[pitch_rank]],
                         markersize = 400*np.mean(dataDict['other'][channel]['amp'][samples_blend])*
                         sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))
    else:
        pitches_blended = np.mean(dataDict['other'][0]['pitch'][:, samples_blend], axis = 1)
        pitch_inds = np.argsort(pitches_blended)
        for pitch_rank in range(-pitchShow, 0):
            plt.plot(1.5, #to handle l/r
                     2.25, #height
                     'o',
                     markerfacecolor = colorList[pitch_inds[pitch_rank]], 
                     markeredgecolor = colorList[pitch_inds[pitch_rank]],
                     markersize = 400*np.mean(dataDict['other'][0]['amp'][samples_blend])*
                     sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))

    plt.xlim(-1.167,  4.167)
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
# if f"{filename}_frames" not in os.listdir():
#     os.mkdir(f"{filename}_frames")

if f"{filename}.avi" in os.listdir():
    os.remove(f"{filename}.avi")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
os.system(f"rm -f {filename}.avi")
image = plotimages(400, dataDict = channels, 
                   colorList = colorList, 
                   splitStereo = splitStereo, 
                   sampBlend = sampBlend, pitchShow = 2)
    
size = (1300, 900)

video = cv2.VideoWriter(f"{filename}.avi", 
                        fourcc, 
                        sr/hop_length, 
                        size)


for i in tqdm(range(len(channels['vocals'][0]['amp']))):
    image = plotimages(i, dataDict = channels, 
                       colorList = colorList, 
                       splitStereo = splitStereo, 
                       sampBlend = sampBlend, pitchShow = 2)
    crop_img = image[144:1056, 200:1400]
    if i == 90:
        cv2.imwrite('test.jpg', crop_img) 
        
    resized = cv2.resize(crop_img, dsize = size, interpolation = cv2.INTER_LINEAR)
    video.write(resized)


video.release()
cv2.destroyAllWindows()
print(datetime.now()-startt)
if f"{filename}_waud.avi" in os.listdir():
    os.remove(f"{filename}_waud.avi")
os.system(f"ffmpeg -i {filename}.avi -i {media_name} -map 0 -map 1:a -c:v copy -shortest {filename}_waud.avi")
