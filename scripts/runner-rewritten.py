import os
import cv2
import torch
import PIL
import datetime
#import autotrain
import actions

cam_sel=0
min_conf=0.35
train_time='00-00-00'
folders=['img_capture','detect_res','recog_res']

os.chdir('..')
this_path=os.path.abspath(os.getcwd())
record_path='{}\\records'.format(this_path)
print('\ncreating folders...')
try:
    os.mkdir(record_path)
except OSError as error:
    print(error)

dt_time=datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
run_path='{}\\run_{}'.format(record_path,dt_time)
os.mkdir(run_path)

file=open('records\\run_logs.txt','a+')
file.write('\n\nawake '+dt_time)

for f in folders:
    os.mkdir(os.path.join(run_path,f))

print('\npreparing models...')
model_detect=torch.hub.load(this_path+'\\yolov5-master','custom',path=this_path+'\\weights\\detect.pt',source='local')
model_detect.conf=min_conf

model_recog=torch.hub.load(this_path+'\\yolov5-master','custom',path=this_path+'\\weights\\recog.pt',source='local')

cam=cv2.VideoCapture(cam_sel)
img_count=0

def capture(frame):
    global img_count,run_path,folders
    curr_img_path=os.path.join(run_path,folders[0])
    #curr_img_path=run_path+'\\{}\\img_{}.jpg'.format(folders[0],img_count)
    cv2.imwrite(curr_img_path+'\\img{}.jpg'.format(img_count),frame)
    return PIL.Image.open(curr_img_path+'\\img{}.jpg'.format(img_count))
    img_count+=1

def detect(img):
    global model_detect,folders
    result=model_detect(img)
    res_pd=result.pandas().xyxy[0]
    if len(res_pd['name'].values):
        result.save(save_dir=run_path+'\\{}'.format(folders[1]))
        return res_pd['name'].values[0],res_pd['confidence'].values[0]
    else:
        return 'none',0

def recog(img):
    global model_recog,folders
    result=model_recog(img)
    res_pd=result.pandas().xyxy[0]
    if len(res_pd['name'].values):
        result.save(save_dir=run_path+'\\{}'.format(folders[2]))
        return res_pd['name'].values[0],res_pd['confidence'].values[0]
    else:
        return 'no one',0

print('\nall done now starting...\n')
while True:
    res_arr=[]
    ret,frame=cam.read()
    if not ret:
        print('cam error')
        break
    else:
        cv2.imshow('cam_view',frame)
        k=cv2.waitKey(125)
        if k==27:
            print('closing')
            break
        elif k==32:
            curr_img=capture(frame)
            dt_time=datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            res_arr.append('\n{} : process started -> img_{}.jpg taken -> '.format(dt_time,img_count-1))
            print('photo taken ..... detection started')
            val,conf=detect(curr_img)
            print('detected:',val,'| confidence:',conf)
            if conf>0:
                res_arr.append('person detected with conf: {} -> '.format(round(conf,3)))
                print('human detected ..... recognition started')
                val,conf=recog(curr_img)
                print('recognised:',val,'| confidence:',conf)
                if conf>0:
                    res_arr.append('access granted to {} with conf {} -> '.format(val,round(conf,3)))
                    print('welcome',val)
                    #actions.pos_action(val)
                else:
                    res_arr.append('access denied -> ')
                    print('you are not registered')
                    #actions.neg_action()
            else:
                res_arr.append('detected no one -> ')
                print('not detected')
            res_arr.append('process terminated')
            file.writelines(res_arr)
            print('\n')


        #adding persons
        elif k==110:
            per_name=input('enter the name of person: ')
            k=cv2.waitKey(125)
            if k==32:
                curr_img=capture(frame)
                val,conf=detect(curr_img)
    
    #if str(datetime.time.strftime('%H-%M-%S'))==train_time:
    #    autotrain.train()

file.close()
cam.release()
cv2.destroyAllWindows()