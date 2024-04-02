import os
import shutil


root = r"D:/workspace/datasets/MMFV-Dataset"
nth = 10
start_idx = 0
out_folder = f"MMFV-{nth}th-start_{start_idx}"
out_root = r"D:/workspace/datasets/" + out_folder

for idx, subject in enumerate(sorted(os.listdir(root))):
    print(f'[{idx}] | subject: {subject}')
    subject_path = os.path.join(root, subject)
    
    for sess in os.listdir(subject_path):
        sess_path = os.path.join(subject_path, sess)
    
        for finger in os.listdir(sess_path):
            finger_path = os.path.join(sess_path, finger)
    
            for mov in os.listdir(finger_path):
                mov_path = os.path.join(finger_path, mov)
                frames_vid1 = [i for i in os.listdir(mov_path) if i.startswith('1_')]
                frames_vid2 = [i for i in os.listdir(mov_path) if i.startswith('2_')]

                # Sort based on frame number
                frames_vid1 = sorted(frames_vid1, key=lambda x: int(x[:-4].split('_')[1]))
                frames_vid2 = sorted(frames_vid2, key=lambda x: int(x[:-4].split('_')[1]))

                # Select every nth frame
                selected_frames = frames_vid1[start_idx::nth] + frames_vid2[start_idx::nth]
                
                # Copy
                for frame in selected_frames:
                    in_path = os.path.join(mov_path, frame)
                    out_path = os.path.join(out_root, subject, sess, finger, mov)
                    os.makedirs(out_path, exist_ok=True)
                    shutil.copy2(in_path, out_path)
