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
import argparse

class Visualizer:
    def __init__(self, 
             media_name, 
             color_palette = 'hsv', 
             hop_length = 512):
        #add white to colorlist for drums
        self.colorList = sns.color_palette(color_palette, 12)
        self.colorList+=[(1,1,1)]
        #balance between low number for precision, high number for stability of pitch/amp calc
        self.hop_length = hop_length
        #load all files, 
        self.media_name = media_name
        self.split_stereo = {'drums':False, 'vocals':True, 'other': True}
        self.samp_blend = {'drums':3, 'vocals':7, 'other': 7, 'bass':5}
        self.filename = self.media_name.split('/')[-1].split('.')[0]
        self.channels = {}


    def load(self):
        # generate separated files with demucs, move to their own folders
    #     if not(os.path.isdir(f"demucs/separated/demucs/{filename}")):
    #         print('Separating sources')
    #         os.system(f'python -m demucs.separate -d cpu --dl raw_music/{media_name}')

        if not(os.path.isdir(f"demucs/separated/demucs/{self.filename}/vocals")):
            for file in os.listdir(f'demucs/separated/demucs/{self.filename}'):
                os.mkdir(f"demucs/separated/demucs/{self.filename}/{file.split('.')[0]}")
                os.system(f"mv demucs/separated/demucs/{self.filename}/{file} demucs/separated/demucs/{self.filename}/{file.split('.')[0]}")

        #load
        print('Loading, splitting stereo audio')

        for i, source in enumerate(['vocals','drums','other','bass']):
        #     print(source)
            self.channels['filename'] = self.filename.split('.')[0]
            self.channels[source] = {}
        #     channels[source]['combined'] = {}
            self.channels[source][0] = {}
            self.channels[source][1] = {}
            if 'left.wav' not in os.listdir(f"demucs/separated/demucs/{self.filename}/{source}"):
        #     split to left/right (how to automate if there is no stereo?)
                os.system(f"ffmpeg -i demucs/separated/demucs/{self.filename}/{source}/{source}.wav -map_channel 0.0.0 demucs/separated/demucs/{self.filename}/{source}/left.wav -map_channel 0.0.1 demucs/separated/demucs/{self.filename}/{source}/right.wav")

        #     channels[source]['combined']['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{source}/{source}.wav", sr=None)
            self.channels[source][0]['audio'], self.sr = librosa.load(f"demucs/separated/demucs/{self.filename}/{source}/left.wav", sr=None)
            self.channels[source][1]['audio'], self.sr = librosa.load(f"demucs/separated/demucs/{self.filename}/{source}/right.wav", sr=None)
        #     channels[source]['stft'] = librosa.stft(channels[source]['audio'])

    def analyze(self):
        # generate pitch and amplitude
        print('Analysis')
        for source in self.channels.keys():
            if source != 'filename':
                for channel in [0,1]:
                    if source != 'drums':
                        chromagram = librosa.feature.chroma_stft(self.channels[source][channel]['audio'], sr=self.sr, hop_length=self.hop_length)
                        self.channels[source][channel]['pitch'] = chromagram
                    else:
                        self.channels[source][channel]['pitch'] = np.repeat(12, int(np.floor(len(self.channels[source][channel]['audio'])/self.hop_length)))

                    self.channels[source][channel]['amp'] = np.array([np.mean(abs(self.channels[source][channel]['audio'][(s*self.hop_length):(s*self.hop_length+self.hop_length)])) 
                                                    for s in range(int(np.floor(len(self.channels[source][channel]['audio'])/self.hop_length))-1)])

    def plotimages(self, s, pitch_show = 2):
        # take maximum-amplitude pitch for the sample, normalize sample ampl by maximum ampl (or by "other"?)
        # for now, use pitch from channel with larger amp. 
        # how to handle two-channel bass better? revert back to circles?

        import matplotlib.patches as mpatches

        channel = np.argmax([abs(self.channels['bass'][0]['amp'][s]),abs(self.channels['bass'][1]['amp'][s])])

        #blend across multiple windowed samples to reduce jitter and flashing
        samples_blend = list(range(max([s-self.samp_blend['bass'], 0]),min([s+self.samp_blend['bass']+1, len(self.channels['bass'][0]['amp'])])))

        #background color = mean of bass pitches across blended samples
    #     modulated by relative amplitude to maximum bass amplitude
    #or normalize to channel[source][channel]['centroid']?
        basscolor = self.colorList
        bgcolor = [c*0.3*np.mean(self.channels['bass'][0]['amp'][samples_blend])/max(self.channels['bass'][0]['amp']) for c in basscolor[np.argmax([np.mean(self.channels['bass'][0]['pitch'][v,samples_blend]) 
                                        for v in range(self.channels['bass'][0]['pitch'].shape[0])])]] 

        fig, ax = plt.subplots(1,1)

        # add a rectangle
        rect = mpatches.Rectangle([-2,-2], 7, 7, edgecolor="none", facecolor = bgcolor)
        ax.add_patch(rect)

        if self.split_stereo['drums']:
            samples_blend = list(range(max([s-self.samp_blend['drums'], 0]),s))

            for channel in [0,1]:
                #drums
                plt.plot(1+channel,2.25, 'o',
                         markerfacecolor = 'none', 
                         markeredgecolor = self.colorList[self.channels['drums'][channel]['pitch'][s]], 
                         markersize = np.mean(self.channels['drums'][channel]['amp'][samples_blend])*250)
        else:
            samples_blend = list(range(max([s-self.samp_blend['drums'], 0]),min([s+self.samp_blend['drums']+1, len(self.channels['drums'][0]['amp'])])))
            plt.plot(1.5,2.25, 'o',
                         markerfacecolor = 'none', 
                         markeredgecolor = self.colorList[self.channels['drums'][0]['pitch'][s]], 
                         markersize = np.mean(self.channels['drums'][0]['amp'][samples_blend])*250)

        #
        if self.split_stereo['vocals']:
            samples_blend = list(range(max([s-self.samp_blend['vocals'], 0]),min([s+self.samp_blend['vocals']+1, len(self.channels['vocals'][0]['amp'])])))

            for channel in [0,1]:
        #grab pitches in reverse order of strength, only using top few
                pitches_blended = np.mean(self.channels['vocals'][channel]['pitch'][:, samples_blend], axis = 1)
                pitch_inds = np.argsort(pitches_blended)
                for pitch_rank in range(-pitch_show, 0):
                    plt.plot(.5+channel*2, #to handle l/r
                             0.75, #height
                             'o',
                             markerfacecolor = 'none', 
                             markeredgecolor = self.colorList[pitch_inds[pitch_rank]],
                             markersize = 400*np.mean(self.channels['vocals'][channel]['amp'][samples_blend])*
                             sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))
        else:
            pitches_blended = np.mean(self.channels['vocals'][0]['pitch'][:, samples_blend], axis = 1)
            pitch_inds = np.argsort(pitches_blended)
            for pitch_rank in range(-pitch_show, 0):
                plt.plot(1.5, #to handle l/r
                         0.75, #height
                         'o',
                         markerfacecolor = 'none', 
                         markeredgecolor = self.colorList[pitch_inds[pitch_rank]],
                         markersize = 400*np.mean(self.channels['vocals'][0]['amp'][samples_blend])*
                         sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))

        if self.split_stereo['other']:
            samples_blend = list(range(max([s-self.samp_blend['other'], 0]),min([s+self.samp_blend['other']+1, len(self.channels['other'][0]['amp'])])))

            for channel in [0,1]:
        #grab pitches in reverse order of strength, only using top few
                pitches_blended = np.mean(self.channels['other'][channel]['pitch'][:, samples_blend], axis = 1)
                pitch_inds = np.argsort(pitches_blended)
                for pitch_rank in range(-pitch_show, 0):
                    plt.plot(.5+channel*2, #to handle l/r
                             1.5, #height
                             'o',
                             markerfacecolor = 'none', 
                             markeredgecolor = self.colorList[pitch_inds[pitch_rank]],
                             markersize = 500*np.mean(self.channels['other'][channel]['amp'][samples_blend])*
                             sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))
        else:
            pitches_blended = np.mean(self.channels['other'][0]['pitch'][:, samples_blend], axis = 1)
            pitch_inds = np.argsort(pitches_blended)
            for pitch_rank in range(-pitch_show, 0):
                plt.plot(1.5, #to handle l/r
                         1.5, #height
                         'o',
                         markerfacecolor = 'none', 
                         markeredgecolor = self.colorList[pitch_inds[pitch_rank]],
                         markersize = 500*np.mean(self.channels['other'][0]['amp'][samples_blend])*
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
    
    def generate_video(self):
        startt = datetime.now()
        # if f"{filename}_frames" not in os.listdir():
        #     os.mkdir(f"{filename}_frames")

        if f"{self.filename}.avi" in os.listdir():
            os.remove(f"{self.filename}.avi")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        size = (1300, 900)

        video = cv2.VideoWriter(f"{self.filename}.avi", 
                                fourcc, 
                                self.sr/self.hop_length, 
                                size)


        for i in tqdm(range(len(self.channels['vocals'][0]['amp']))):
            image = self.plotimages(i, pitch_show = 3)
            crop_img = image[144:1056, 200:1400]

            resized = cv2.resize(crop_img, dsize = size, interpolation = cv2.INTER_LINEAR)
            video.write(resized)


        video.release()
        cv2.destroyAllWindows()
        print(datetime.now()-startt)
        if f"{self.filename}_final.avi" in os.listdir():
            os.remove(f"{self.filename}_final.avi")
        os.system(f"ffmpeg -i {self.filename}.avi -i {self.media_name} -map 0 -map 1:a -c:v copy -shortest {self.filename}_final.avi")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process filenames.')
    parser.add_argument('filenames', metavar='f', type=str,
                        help='paths to audio files for parsing')

    args = parser.parse_args()
    print(args.filenames)
    viz = Visualizer(args.filenames)
    viz.load()
    viz.analyze()
    viz.generate_video()
            



