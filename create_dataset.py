import pyautogui
import time
import cv2

number_imgs = 1000

for imgnum in range(number_imgs):
    print('Collecting image {}'.format(imgnum))

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    screenshot = pyautogui.screenshot()
    screenshot.save(f'D:\Projects\CS-GOObjectDetection\datasets\coco\images\ToSort\\{timestamp}' + '.jpg')
    
    #imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
    time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break