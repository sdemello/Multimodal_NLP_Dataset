import os
import sys
import pandas as pd
import assemblyai as aai
import cv2
from Find_Timestamps_Scene_Changes import *

def get_clip_duration(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s = round(frame_count / fps, 2)
    video.release()
    return duration_s

aai.settings.api_key = "07158802fd464ca197307f185e84f9f3"
folder = "C:/Users/User/OneDrive/Documents/Research Project Data - Copy/"

list1 = []
i = 0
trim_to_longest_scene = True
for file in os.listdir(folder):
    print(f"{i+1}) {file}")
    temp_file_path = os.path.join(folder, file)
    if trim_to_longest_scene is True:
        start_time, end_time = find_timestamps_longest_scene(temp_file_path)
        print(start_time, end_time)
        config = aai.TranscriptionConfig(speaker_labels=True, audio_start_from=start_time*1000, audio_end_at=end_time*1000)
    else:
        config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    temp_transcript = transcriber.transcribe(temp_file_path, config=config)
    temp_transcript_text = temp_transcript.text
    video_duration = get_clip_duration(temp_file_path)
    row = [file, video_duration, temp_transcript_text]
    print(row)
    list1.append(row)
    i += 1

dataframe1 = pd.DataFrame(list1, columns=["Video File Name", "Video Duration", "Transcript Text"])
if trim_to_longest_scene is True:
    export_file = "C:/Users/User/Downloads/ResearchProjectTrimmedText.csv"
else:
    export_file = "C:/Users/User/Downloads/ResearchProjectText.csv"
dataframe1.to_csv(export_file)




