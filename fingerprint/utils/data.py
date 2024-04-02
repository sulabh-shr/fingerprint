import os
from natsort import natsorted


def iterate_mmfv_files(root, start_frame=0, every_n_frame=1, out_root=None):
    subjects = natsorted(os.listdir(root))

    for idx, subject in enumerate(subjects):
        subject_path = os.path.join(root, subject)
        if not os.path.isdir(subject_path):
            continue
        for sess in os.listdir(subject_path):
            sess_path = os.path.join(subject_path, sess)
            for finger in os.listdir(sess_path):
                finger_path = os.path.join(sess_path, finger)
                for mov in os.listdir(finger_path):
                    mov_path = os.path.join(finger_path, mov)
                    frames_vid1 = natsorted([i for i in os.listdir(mov_path) if i.startswith('1_')])
                    frames_vid2 = natsorted([i for i in os.listdir(mov_path) if i.startswith('2_')])

                    # Select every nth frame
                    selected_frames = frames_vid1[start_frame::every_n_frame] + frames_vid2[start_frame::every_n_frame]
                    for frame in selected_frames:
                        img_path = os.path.join(mov_path, frame)
                        result = {
                            'subject': subject,
                            'sess': sess,
                            'finger': finger,
                            'mov': mov,
                            'frame': frame,
                            'mov_path': mov_path,
                            'img_path': img_path,
                            'class': f'{subject}-{finger}'
                        }
                        if out_root is not None:
                            out_dir = os.path.join(out_root, subject, sess, finger, mov)
                            out_path = os.path.join(out_dir, frame)
                            result['out_dir'] = out_dir
                            result['out_path'] = out_path
                        yield result


if __name__ == '__main__':
    _root = r"D:\workspace\datasets\MMFV-25th"
    for i in iterate_mmfv_files(_root):
        print(i['subject'], i['sess'], i['finger'], i['mov'])
