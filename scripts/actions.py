import cv2
import torch
import os
import time
import telepot

#token and receiver id
token='5765245671:AAEx_x05L-apT1_nECw9HAvkNLUx19kiLIw'
receiver_id = 853458661 ##https://api.telegram.org/bot<token>/getupdates
bot =telepot.Bot(token)
bot.sendMessage(receiver_id,'Your camera is active now.')

def pos_action(predictor, vis_folder, current_time, args):
    #actions executed if recognised
    
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            print('pos_works')
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if outputs[0] is not None:
                bot.sendMessage(receiver_id, "Access Granted")
                #filename = "D:\savedImage.jpg"
                cv2.imwrite(filename, frame)
                bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                os.remove(filename)
            cv2.imshow("Image", result_frame)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
   

def neg_action():
    #actions executed otherwise
    print('neg_works')
    return
    
    
    
    