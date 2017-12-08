import numpy
import re
import os
import sys
import PIL.Image
import argparse

'''

Builds a numpy array from annotation and images that can be processes by YAD2k

'''



classes_dict = {"aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4, "bus":5, "car":6, "cat":7, "chair":8, "cow":9, "diningtable":10, "dog":11, "horse":12, "motorbike":13, "person":14, "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19}



def convert_annotation(image_id):
    '''
    Reads annotations from a file and returns then a numpy array of numpy arrays
    A nested array for each bountding box

    Example

    [[class, x_1, y_1, x_2, y_2]]
    '''
    cleaned_boxes = []
    
    file = open('train/{}'.format(image_id))
 
    lines = file.readlines()
    for line in lines:
        blob , i = str(line).split(".jpg_")
        ''' discard blob, is not needed'''
        parts = i.partition(',')
    
        if parts[2] not in "1 0\n": 
     
            strings = parts[2].split(" ")
  
            nums = [] 
            for item in strings:
                #if len(item) is not 0:
                s = re.sub("[\s+]", "", item)
                s = re.sub(" ", "", item)
                s = re.sub("  ", "", item)
                s = re.sub("\n", "", item)
                try: 

                    nums.append(int(s.strip()))
                except Exception as e:
                    continue
                    # print(e)
                    # print("empty string {}".format(s))


            nums_paried = []


            for i in range((len(nums) //2)):
                nums_paried.append([nums[i * 2] , nums[i* 2 +1]])

            y_1 = [1]
            x_1 = [1]
            y_2 = [1]
            x_2 = [1]


            boxes = {}

            for pair in nums_paried:
           
                if pair[1] in boxes:
                    boxes[pair[1]].append(pair)
                else:
                    boxes[pair[1]] = []
                    boxes[pair[1]].append(pair)


            for key, value in boxes.items():
                y_1 = value[0][0] // 400
                y_2 = value[len(value) - 2][0] // 400
                x_1 = value[0][0] % 400
                x_2 = x_1 + value[0][1]

                #print("class {}, x_1 {}, x_2 {}, y_1 {}, y_2 {}".format(classes_dict[parts[0]],x_1,x_2,y_1,y_2))
                box = [classes_dict[parts[0]], x_1, y_1, y_2 - y_1, x_2 - x_1]
                box = numpy.array(box)
                
                cleaned_boxes.append(box)

    
    return numpy.array(cleaned_boxes)

def generate_set(begin, end, output_file):


    files = os.listdir("train/")

    files.sort()

    text_files = []
    image_files = []


    for i in range(begin, int(end)):
        if ".jpg" in files[i]:
            image_files.append(files[i])
        else:
            text_files.append(files[i])


    '''
    Check that the image and annotation files match

    '''
    assert len(image_files), len(text_files)


    for i in range(len(image_files)):
        image_name = re.sub('.jpg', '',image_files[i])
        file_name = re.sub('.txt','',text_files[i])

        if image_name not in file_name:
            print("annotation: {} and image file: {} does not match".format(file_name, image_name))

    annotations = []

    print(len(image_files), len(text_files))

    for file in text_files:
        annotations.append(convert_annotation(file))

    image_labels = numpy.array(annotations)


    images = []

    for image in image_files:
        img = numpy.array(PIL.Image.open(os.path.join('train', image )).resize((640, 480)).convert('RGB'), dtype=numpy.uint8)
        images.append(img)

    images = numpy.array(images, dtype=numpy.uint8)


    print("Dataset contains {} images".format(images.shape[0]))
    numpy.savez(os.path.join('numpy_arrays', output_file ), images=images, boxes=image_labels)


'''

Set variables to determine which files to collect build into numpy array

'''
parser = argparse.ArgumentParser(description='Generate a training set')
parser.add_argument('--begin', metavar='-b', type=int, help='pick an even number')
parser.add_argument('--end', metavar='-e', type=int, help='Pick an even number')
parser.add_argument('--name', metavar='-n', help='File Name')
args = parser.parse_args()

generate_set(args.begin, args.end, args.name)



# print(convert_annotation("2007_006661.txt"))
