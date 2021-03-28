python.exe -m demucs.separate -d cpu --dl rome.m4a
python run visualizer.py
ffmpeg -i rome.avi -i rome.m4a -map 0 -map 1:a -c:v copy -shortest rome_waud.avi