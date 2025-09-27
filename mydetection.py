from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#image_input = jetson.utils.loadImage("/home/nvidia/jetson-inference/data/images/fruit_0.jpg")
camera = jetson.utils.videoSource("/home/nvidia/jetson-inference/data/images")
display = jetson.utils.videoOutput("/home/nvidia/jetson-inference/data/images/fruit_0.jpg")
#result_saver = saveImage("/Home/nvidia/jetson-inference/data/images/result/output.jpg")

while display.IsStreaming():
    img = camera.Capture()
    if img is None:
        continue
    detections = net.Detect(img)
    print(detections)
    
# img = jetson.utils.loadImage("/home/nvidia/jetson-inference/data/images/fruit_0.jpg")
# if img is None:
#     print("Error")
# else:
#     detections = net.Detect(img)

    #target_class_id = 54
    #print("(ClassID={}".format(target_class_id))
    #for det in detections:
        #if det.ClassID == target_class_id:
        #print("ClassID:{}".format(det.ClassID))
        #print("Confidence:{:.4f}".format(det.Confidence))
        #print("Left:{:.4f},Top:{:.4f}".format(det.Left, det.Top))
        #print("Right:{:.4f},Bottom:{:.4f}".format(det.Right, det.Bottom))
        # print("Width:{:.4f},Height:{:.4f}".format(det.Width, det.Height))
        # print("Area:{:.4f}".format(det.Area))
        # print("Center:{:.4f}, {:.4f}".format(det.Center[0], det.Center[1]))
        

    display.Render(img)
    #result_saver.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    #result_saver.SetStatus("Result Saved | Network {:.0f} FPS".format(net.GetNetworkFPS()))
