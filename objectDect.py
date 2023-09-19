import numpy as np
import cv2
import datetime

image_path = '\\img\\car-road.jpg'
prototxt_path = 'ML\\Object Detector\\Data\\MobileNetSSD_deploy.prototxt.txt'
model_path = 'ML\\Object Detector\\Data\\MobileNetSSD_deploy.caffemodel'
min_confidance = 0.2

classes = ["background","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","chair","cow","diningtable",
           "dog","horse","motorbike","person","pottedplant","sheep",
           "sofa","train","tvmonitor"]

np.random.seed(543210)
colors = np.random.uniform(0,255,size=(len(classes),3))
net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)

#image = cv2.imread(image_path)
# cap = cv2.VideoCapture('ML\\Object Detector\\img\\edmonton_canada.mp4')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
if (cap.isOpened()==False):
     print("Error occured video stream or File")

#list for data base With time
Mongo = [] 
while (cap.isOpened()):                 
        ret,image = cap.read()
        if ret == True:

            #print(type(image))
            #print(image.shape[1])

            height,widht = image.shape[0],image.shape[1]
            blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007,(300,300),130)

            net.setInput(blob)
            detected_objects = net.forward()

            for i in range(detected_objects.shape[2]):

                confidence = detected_objects[0][0][i][2]

                if confidence > min_confidance:

                    class_index = int (detected_objects[0,0,i,1])

                    upper_left_x = int(detected_objects[0,0,i,3]*widht)
                    upper_left_y = int(detected_objects[0,0,i,4]*height)
                    lower_right_x = int(detected_objects[0,0,i,5]*widht)
                    lower_right_y = int(detected_objects[0,0,i,6]*height)

                    predection_text = f"{classes[class_index]}:{confidence:.2f}%"
        
                    cv2.rectangle(image,(upper_left_x,upper_left_y),(lower_right_x,lower_right_y),colors[class_index],3)    # type: ignore
                    cv2.putText(image,predection_text,(upper_left_x,upper_left_y - 15 if upper_left_y >30 else upper_left_y + 15),cv2.FONT_HERSHEY_SIMPLEX,0.6,colors[class_index],2)

                    Mongo.append(classes[class_index])

            cv2.imshow("Detected Objects",image)

            if cv2.waitKey(25) & 0xFF == ord('q'):  
                    print("Dectacted items are :")
                    now = datetime.datetime.now()
                    Curr_time = now.strftime("%B/%d/%Y %H:%M:%S")
                    data_db_set = set(Mongo)
                    data_db_list = list(data_db_set)
                    data_db_list.insert(0,Curr_time)
                    print(data_db_list)
                    break
        else:
            break

cap.release() 
cv2.destroyAllWindows()
                
