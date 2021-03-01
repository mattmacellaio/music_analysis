import numpy as np
import librosa, sklearn
import matplotlib.pyplot as plt
import librosa.display
from datetime import datetime
import cv2
import os
from tqdm import tqdm
import seaborn as sns
import argparse

plt.rcParams['figure.figsize'] = [16, 12]


class Visualizer:
    def __init__(self, media_name, color_palette='hsv', hop_length=1024):
        # add white to colorlist for drums
        self.colorList = sns.color_palette(color_palette, 12)
        self.colorList += [(1, 1, 1)]
        # balance between low number for precision, high number for stability of pitch/amp calc
        self.hop_length = hop_length
        # load all files,
        self.media_name = media_name
        self.split_stereo = {'drums': False, 'vocals': False, 'other': False}
        self.samp_blend = {'drums': 3, 'vocals': 7, 'other': 7, 'bass': 10}
        self.filename = self.media_name.split('/')[-1].split('.')[0]
        self.channels = {}

    def load(self):
        # generate separated files with demucs, move to their own folders
        #if not(os.path.isdir(f"demucs/separated/demucs/{self.filename}")):
        #    print('Separating sources')
        #    os.system(f'python -m demucs/demucs.separate -d cpu --dl raw_music/{self.media_name}')

        filepath = f'demucs/separated/demucs/{self.filename}'
        if not (os.path.isdir(f"{filepath}/vocals")):
            files = os.listdir(filepath)
            for file in files:
                os.mkdir(f"{filepath}/{file.split('.')[0]}")
                os.system(
                    f"mv {filepath}/{file} {filepath}/{file.split('.')[0]}")

        # load
        print('Loading, splitting stereo audio')

        for i, source in enumerate(['vocals', 'drums', 'other', 'bass']):
            #     print(source)
            self.channels['filename'] = self.filename.split('.')[0]
            self.channels[source] = {}
            #     channels[source]['combined'] = {}
            self.channels[source][0] = {}
            self.channels[source][1] = {}
            if 'left.wav' not in os.listdir(f"{filepath}/{source}"):
                #     split to left/right (how to automate if there is no stereo?)
                os.system(
                    f"ffmpeg -i {filepath}/{source}/{source}.wav -map_channel 0.0.0 {filepath}/{source}/left.wav -map_channel 0.0.1 {filepath}/{source}/right.wav")

            #     channels[source]['combined']['audio'], sr = librosa.load(f"demucs/separated/demucs/{filename}/{source}/{source}.wav", sr=None)
            self.channels[source][0]['audio'], self.sr = librosa.load(
                f"{filepath}/{source}/left.wav", sr=None)
            self.channels[source][1]['audio'], self.sr = librosa.load(
                f"{filepath}/{source}/right.wav", sr=None)
        #     channels[source]['stft'] = librosa.stft(channels[source]['audio'])

    def analyze(self):
        # generate pitch and amplitude
        print('Analysis')
        for source in self.channels.keys():
            if source != 'filename':
                for channel in [0, 1]:
                    n_fft = 4096
                    #get frequencies
                    freqs = librosa.fft_frequencies(self.sr, n_fft)
                    C = 261.626
                    #half-steps from middle C
                    hs = 12*np.log(freqs/ C) / np.log(2)
                    pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

                    pitch = [pitches[np.mod(int(hs), 12)] for i, hs in enumerate(np.round(hs[1:]))]
                    octave = [str(int(np.log2(freq/C)+4)) for freq in freqs[1:]]
                    #future steps: drop or combine duplicates
                    self.channels[source][channel]['pitch'] = [pitch[i]+octave[i] for i in range(len(pitch))]

                    #get amplitude for all freqs but first one, since first freq is undefinable above
                    self.channels[source][channel]['amp'] = librosa.amplitude_to_db(np.abs(librosa.stft(self.channels[source][channel]['audio'], hop_length = self.hop_length, n_fft = n_fft)))[1:, :]

    def plotimages(self, s):
        ## new plotting
        
        #working but need to add loop, timing, change mask darkness based on brightness

        import cairo
        import numpy as np
        from IPython.display import Image, display
        from math import pi
        from io import BytesIO
        import colorsys

        n_fft = 4096
        audio, sr = librosa.load(f"demucs/separated/demucs/anastasia/other/left.wav", sr=None)

        stft= librosa.stft(audio, hop_length = 512, n_fft = n_fft)

        freqs = librosa.fft_frequencies(sr, n_fft)
        use_inds = [i for i,x in enumerate(freqs) if ((x<4000) & (x>0))]
        freqs = freqs[use_inds]
        C = 261.626
        hs = 12*np.log(freqs/ C) / np.log(2)

        pitches = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        pitch = [pitches[np.mod(int(h), 12)]+str(int((np.log2(freqs[i]/C)+4))) for i, h in enumerate(np.round(hs))]

        # use_pitches = [pitch[i] for i in use_inds]
        stft_use = librosa.amplitude_to_db(np.abs(stft[use_inds,:]))

        def disp(draw_func):
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 400, 400)
            ctx = cairo.Context(surface)
            draw_func(ctx, 400, 400)
            with BytesIO() as fileobj:
                surface.write_to_png(fileobj)
                display(Image(fileobj.getvalue(), width=400))

        @disp

        def draw(cr, width, height):
            #scriabin colors from http://www.flutopedia.com/sound_color.htm - perhaps too blue-skewed?
        #balanced colors from https://medium.com/@jeremygeltman/octaves-of-light-3c153d328a4b
        # test colors to see what evenly-spaced hues look like

            colors = {'scriabin':
                      {'C': (0.996,0,0), 
                      'G': (1, 0.498, 0), 
                      'D': (1,1,0), 
                      'A': (0.2, 0.8, 0.2), 
                      'E': (0.761, 0.949, 0.933), 
                      'B': (0.549, 0.784, 0.996), 
                      'F#': (0.498, 0.541, 0.992),
                      'C#': (0.565, 0, 1),
                      'G#': (0.733, 0.459, 0.988), 
                      'D#': (0.722, 0.275, 0.545), 
                      'A#': (0.659, 0.404, 0.482), 
                      'F': (0.671, 0, 0.204)}, 
                      'balanced': 
                      {'C': (59.7, 100, 60), 
                      'G': (45.2, 98.4, 49.2), 
                      'D': (36.4, 98.4, 49.6), 
                      'A': (18.4, 98.4, 51.2), 
                      'E': (5.3, 99.2, 53.3), 
                      'B': (338.9, 74, 37.6), 
                      'F#': (285.6,98.9,34.7),
                      'C#': (262.5,100,32),
                      'G#': (223.6,99.2,50.2),
                      'D#': (197.7,97.1,41),
                      'A#': (95.5,55.1,44.5),
                      'F': (67.9,81.9,54.5)}, 
                      'test':{'C': (60, 100, 60), 
                      'G': (90, 100, 60), 
                      'D': (120, 100, 60), 
                      'A': (150, 100, 60), 
                      'E': (180, 100, 60), 
                      'B': (210, 100, 60), 
                      'F#': (240, 100, 60), 
                      'C#': (270, 100, 60), 
                      'G#': (300, 100, 60), 
                      'D#': (330, 100, 60), 
                      'A#': (0, 100, 60), 
                      'F': (30, 100, 60)}}

            cr.scale(width, height)
            cr.set_line_width(0.04)

            x, y = 0.5, 0.5
            radius = .5

            colormap = 'test'          
            for d in range(len(use_pitches)):
                pitch_rgb = colors[colormap][np.random.choice(list(colors[colormap].keys()))]            
                octave = int(use_pitches[d][-1])/10
                if colormap == 'scriabin':
                    cr.set_source_rgba(min(pitch_rgb[0]*octave, 1), 
                                       min(pitch_rgb[1]*octave, 1), 
                                       min(pitch_rgb[2]*octave, 1), 
                                       1) #r,g, b, alpha
                else:
                    rgb = colorsys.hls_to_rgb(pitch_rgb[0]/100,octave, 1) #hue lightness saturation in [0,1] space
                    cr.set_source_rgba(rgb[0], rgb[1], rgb[2], 1) #r,g, b, alpha
                cr.set_line_width(0.005)
                cr.move_to(x, y)
                cr.line_to(x+radius*np.cos(d*(2*pi/360)), y+radius*np.sin(d*(2*pi/360)))
                cr.stroke()
            cr.close_path()

            cr.set_source_rgba(0,0,0, 1) #r,g, b, alpha
            mask = cairo.RadialGradient(0.5, 0.5, .3, 0.5, 0.5, .5) #x, y, radius, x2, y2, radius
            mask.add_color_stop_rgba(0, 0, 0, 0, 0.1) #offset, red, green, blue, alpha)
            mask.add_color_stop_rgba(1, 0, 0, 0, 1) #offset, red, green, blue, alpha)
            cr.mask(mask)


        
        
        
        # take maximum-amplitude pitch for the sample, normalize sample ampl by maximum ampl (or by "other"?)
        # for now, use pitch from channel with larger amp. 
        # how to handle two-channel bass better? revert back to circles?

        import matplotlib.patches as mpatches

        channel = np.argmax([abs(self.channels['bass'][0]['amp'][s]), abs(self.channels['bass'][1]['amp'][s])])

        # blend across multiple windowed samples to reduce jitter and flashing
        samples_blend = list(range(max([s - self.samp_blend['bass'], 0]),
                                   min([s + self.samp_blend['bass'] + 1, len(self.channels['bass'][0]['amp'])])))

        # background color = mean of bass pitches across blended samples
        #     modulated by relative amplitude to maximum bass amplitude
        # or normalize to channel[source][channel]['centroid']?
        basscolor = self.colorList
        bgcolor = [
            c * 0.3 * np.mean(self.channels['bass'][channel]['amp'][samples_blend]) / max(self.channels['bass'][channel]['amp']) for
            c in basscolor[np.argmax([np.mean(self.channels['bass'][channel]['pitch'][v, samples_blend])
                                      for v in range(self.channels['bass'][channel]['pitch'].shape[0])])]]

        fig, ax = plt.subplots(1, 1)

        # add a rectangle
        rect = mpatches.Rectangle([-2, -2], 7, 7, edgecolor="none", facecolor=[0,0,0])
        ax.add_patch(rect)

        if self.split_stereo['drums']:
            samples_blend = list(range(max([s - self.samp_blend['drums'], 0]), s))

            for channel in [0, 1]:
                # drums
                plt.plot(1 + channel, 2.25, 'o',
                         markerfacecolor='none',
                         markeredgecolor=self.colorList[self.channels['drums'][channel]['pitch'][s]],
                         markersize=np.mean(self.channels['drums'][channel]['amp'][samples_blend]) * 250)
        else:
            samples_blend = list(range(max([s - self.samp_blend['drums'], 0]),
                                       min([s + self.samp_blend['drums'] + 1, len(self.channels['drums'][0]['amp'])])))
            plt.plot(1.5, 2.25, 'o',
                     markerfacecolor='none',
                     markeredgecolor=self.colorList[self.channels['drums'][0]['pitch'][s]],
                     markersize=np.mean(self.channels['drums'][0]['amp'][samples_blend]) * 250)

        #
        if self.split_stereo['vocals']:
            samples_blend = list(range(max([s - self.samp_blend['vocals'], 0]), min(
                [s + self.samp_blend['vocals'] + 1, len(self.channels['vocals'][0]['amp'])])))

            for channel in [0, 1]:
                # grab pitches in reverse order of strength, only using top few
                pitches_blended = np.mean(self.channels['vocals'][channel]['pitch'][:, samples_blend], axis=1)
                pitch_inds = np.argsort(pitches_blended)
                for pitch_rank in range(-pitch_show, 0):
                    plt.plot(.5 + channel * 2,  # to handle l/r
                             0.75,  # height
                             'o',
                             markerfacecolor='none',
                             markeredgecolor=self.colorList[pitch_inds[pitch_rank]],
                             markersize=400 * np.mean(self.channels['vocals'][channel]['amp'][samples_blend]) *
                                        sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))
        else:
            pitches_blended = np.mean(self.channels['vocals'][0]['pitch'][:, samples_blend], axis=1)
            pitch_inds = np.argsort(pitches_blended)
            for pitch_rank in range(-1, 0):
                plt.plot(1.5,  # to handle l/r
                         1.5,  # height
                         'o',
                         markerfacecolor='none',
                         markeredgewidth = 2,
                         markeredgecolor=self.colorList[pitch_inds[pitch_rank]],
                         markersize=400 * np.mean(self.channels['vocals'][0]['amp'][samples_blend]) *
                                    sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))

        if self.split_stereo['other']:
            samples_blend = list(range(max([s - self.samp_blend['other'], 0]),
                                       min([s + self.samp_blend['other'] + 1, len(self.channels['other'][0]['amp'])])))

            for channel in [0, 1]:
                # grab pitches in reverse order of strength, only using top few
                pitches_blended = np.mean(self.channels['other'][channel]['pitch'][:, samples_blend], axis=1)
                pitch_inds = np.argsort(pitches_blended)
                for pitch_rank in range(-pitch_show, 0):
                    plt.plot(.5 + channel * 2,  # to handle l/r
                             1.5,  # height
                             'o',
                             markerfacecolor='none',
                             markeredgewidth = 2,
                             markeredgecolor=self.colorList[pitch_inds[pitch_rank]],
                             markersize=400 * np.mean(self.channels['other'][channel]['amp'][samples_blend]) *
                                        sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))
        else:
            pitches_blended = np.mean(self.channels['other'][0]['pitch'][:, samples_blend], axis=1)
            pitch_inds = np.argsort(pitches_blended)
            for pitch_rank in range(-pitch_show, 0):
                plt.plot(1.5,  # to handle l/r
                         0.75,  # height
                         'o',
                        markeredgewidth = 2,
                        markerfacecolor='none',
                         markeredgecolor=self.colorList[pitch_inds[pitch_rank]],
                         markersize=500 * np.mean(self.channels['other'][0]['amp'][samples_blend]) *
                                    sum(pitches_blended[pitch_inds[range(pitch_rank, 0)]]))

        plt.xlim(-1.167, 4.167)
        plt.ylim(0, 3)
        plt.axis('off')

        # redraw the canvas
        fig.canvas.draw()

        #     # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                            sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        #     # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
                                self.sr / self.hop_length,
                                size)

        for i in tqdm(range(len(self.channels['vocals'][0]['amp']))):
            image = self.plotimages(i, pitch_show=2)
            crop_img = image[144:1056, 200:1400]

            resized = cv2.resize(crop_img, dsize=size, interpolation=cv2.INTER_LINEAR)
            video.write(resized)

        video.release()
        cv2.destroyAllWindows()
        print(datetime.now() - startt)
        if f"{self.filename}_final.avi" in os.listdir():
            os.remove(f"{self.filename}_final.avi")
        os.system(
            f"ffmpeg -i {self.filename}.avi -i {self.media_name} -map 0 -map 1:a -c:v copy -shortest {self.filename}_final.avi")


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
