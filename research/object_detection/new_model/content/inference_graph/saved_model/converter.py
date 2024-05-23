# import tensorflow as tf

# # Convert the model
# converter = tf.lite.TFLiteConverter.from_saved_model('./') # path to the SavedModel directory
# converter.experimental_enable_resource_variables = True
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 
# converter._experimental_lower_tensor_list_ops = False
# tflite_model = converter.convert()

# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)


import tensorflow as tf
import numpy as np
import cv2
import pathlib

interpreter = tf.contrib.lite.Interpreter(model_path="./new_model/content/inference_graph/saved_model/model.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

interpreter.allocate_tensors()

def draw_rect(image, box):
    y_min = int(max(1, (box[0] * image.height)))
    x_min = int(max(1, (box[1] * image.width)))
    y_max = int(min(image.height, (box[2] * image.height)))
    x_max = int(min(image.width, (box[3] * image.width)))
    
    # draw a rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

for file in pathlib.Path('./images/test/').iterdir():

    if file.suffix != '.jpg' and file.suffix != '.png':
        continue
    
    img = cv2.imread(r"{}".format(file.resolve()))
    new_img = cv2.resize(img, (512, 512))
    interpreter.set_tensor(input_details[0]['index'], [new_img])

    interpreter.invoke()
    rects = interpreter.get_tensor(
        output_details[0]['index'])

    scores = interpreter.get_tensor(
        output_details[2]['index'])
    
    for index, score in enumerate(scores[0]):
        if score > 0.5:
          draw_rect(new_img,rects[0][index])
          
    cv2.imshow("image", new_img)
    cv2.waitKey(0)