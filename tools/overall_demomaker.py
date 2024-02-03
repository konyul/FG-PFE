import sys
def gen_video(scene_name):
    import cv2
    import os
    sensors = os.listdir(scene_name)
    image_folder = scene_name
    
    for sensor in sensors:
        video_name = f'{scene_name}_{sensor}.avi'
        images = [img for img in sorted(os.listdir(os.path.join(image_folder,sensor))) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, sensor, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 1, (width,height))

        for image in images:

            video.write(cv2.imread(os.path.join(image_folder, sensor, image)))

        cv2.destroyAllWindows()
        video.release()
if __name__ == '__main__':
    gen_video(sys.argv[1])
    #gen_video()