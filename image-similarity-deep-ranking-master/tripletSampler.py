import os
import re
import argparse
import numpy as np
from shutil import copyfile
import random

imgs_dir="G:\\LicentaSimona\\dataset\\scrapping\\"
imgs_dir2="G:\\LicentaSimona\\dataset\\cropsScrapping\\"
imgs_no_background="G:\\LicentaSimona\\dataset\\training\\big_sets\\all_white_background_dress\\"
imgs_scrapping="G:\\LicentaSimona\\dataset\\scrapping\\"
triplets_scrapping="G:\\LicentaSimona\\tripletsScrapping.csv"

def move_imgs(src,dest):
        for file in os.listdir(src):
            img_src=src+file
            if(file.find("_crop0")!=-1):
                file_r = file.replace("_crop0","")
            if(file.find("_crop1") != -1):
                file_r = file.replace("_crop1", "")
            if(file.find("_crop2") != -1):
                file_r = file.replace("_crop2", "")
            if(file.find("_crop3") != -1):
                file_r = file.replace("_crop3", "")
            img_dest = src+file_r
            copyfile(img_src, img_dest)

move_imgs(imgs_dir2, imgs_dir2)

def rename_imgs_from_scrapping(src_imgs_directory, dest_imgs_directory,neg_images):
    for dir in os.listdir(src_imgs_directory):
        dir_name=dir
        count=0
        for file in os.listdir(src_imgs_directory+"\\"+dir_name):
            src=src_imgs_directory+"\\"+dir_name+"\\"+file
            if count==0:
                print(file)
                query=dest_imgs_directory+"\\"+dir_name+"\\"+"0_"+dir_name+"_q"+".jpg"
                copyfile(src, query)
                count = count + 1
            else:
                count=count+1
                pos = dest_imgs_directory + "\\"+dir_name+"\\"+ str(count)+"_" + dir_name+"_p_.jpg"
                copyfile(src, pos)
                random_img = random.choice(os.listdir(neg_images))
                print(random_img)
                dest_neg = dest_imgs_directory + "\\"+dir_name+"\\"+ str(count)+"_" + dir_name+"_n_.jpg"
                copyfile(neg_images+random_img, dest_neg)

def generate_triplets_from_scrapping(src_imgs_directory, triplets_file):
    f = open(triplets_file, "a")
    for dir in os.listdir(src_imgs_directory):
        dir_name=dir
        count=0
        query = ""
        for file in os.listdir(src_imgs_directory+"\\"+dir_name):
            if count==0:
                query="0_"+dir_name+"_q"+".jpg"
                pos = str(count) + "_" + dir_name + "_p_.jpg"
                neg = str(count) + "_" + dir_name + "_n_.jpg"
                print(query,pos,neg)
                f.write(query+","+pos+","+neg+"\n")
                count = count + 1
            else:
                count=count+1
                pos = str(count) + "_" + dir_name + "_p_.jpg"
                neg = str(count) + "_" + dir_name + "_n_.jpg"
                f.write(query + "," + pos + "," + neg + "\n")
    f.close()


#generate_triplets_from_scrapping(imgs_dir2, triplets_scrapping)

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def get_negative_images(all_images,image_names,num_neg_images):
    random_numbers = np.arange(len(all_images))
    np.random.shuffle(random_numbers)
    if int(num_neg_images)>(len(all_images)-1):
        num_neg_images = len(all_images)-1
    neg_count = 0
    negative_images = []
    for random_number in list(random_numbers):
        if all_images[random_number] not in image_names:
            negative_images.append(all_images[random_number])
            neg_count += 1
            if neg_count>(int(num_neg_images)-1):
                break
    return negative_images

def get_positive_images(image_name,image_names,num_pos_images):
    random_numbers = np.arange(len(image_names))
    np.random.shuffle(random_numbers)
    if int(num_pos_images)>(len(image_names)-1):
        num_pos_images = len(image_names)-1
    pos_count = 0
    positive_images = []
    for random_number in list(random_numbers):
        if image_names[random_number]!= image_name:
            positive_images.append(image_names[random_number])
            pos_count += 1 
            if int(pos_count)>(int(num_pos_images)-1):
                break
    return positive_images

def triplet_sampler(directory_path, output_path,num_neg_images,num_pos_images):
    classes = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    all_images = []
    for class_ in classes:
        all_images += (list_pictures(os.path.join(directory_path,class_)))
    triplets = []
    for class_ in classes:
        image_names = list_pictures(os.path.join(directory_path,class_))
        for image_name in image_names:
            image_names_set = set(image_names)
            query_image = image_name
            positive_images = get_positive_images(image_name,image_names,num_pos_images)
            for positive_image in positive_images:
                negative_images = get_negative_images(all_images,set(image_names),num_neg_images)
                for negative_image in negative_images:
                    triplets.append(query_image+',')
                    triplets.append(positive_image+',')
                    triplets.append(negative_image+'\n')
            
    f = open(os.path.join(output_path,"triplets.txt"),'w')
    f.write("".join(triplets))
    f.close()


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--input_directory', 
                       help='A argument for input directory')

    parser.add_argument('--output_directory', 
                       help='A argument for output directory')

    parser.add_argument('--num_pos_images', 
                       help='A argument for the number of Positive images per Query image')

    parser.add_argument('--num_neg_images', 
                       help='A argument for the number of Negative images per Query image')                                              
    
    args = parser.parse_args()

    if args.input_directory == None:
        print('Input Directory path is required!')
        quit()
    elif args.output_directory == None:
        print('Output Directory path is required!')
        quit()
    elif args.num_pos_images == None:
        print('Number of Positive Images is required!')
        quit()
    elif args.num_neg_images == None:
        print('Number of Negative Images is required!')
        quit()
    elif int(args.num_neg_images) < 1:
        print('Number of Negative Images cannot be less than 1!')
    elif int(args.num_pos_images) < 1:
        print('Number of Positive Images cannot be less than 1!')

    if not os.path.exists(args.input_directory):
            print (args.input_directory+" path does not exist!")
            quit()

    if not os.path.exists(args.output_directory):
            print (args.input_directory+" path does not exist!")
            quit()

    print ("Input Directory: "+args.input_directory)
    print ("Output Directory: "+args.output_directory)
    print ("Number of Positive image per Query image: "+args.num_pos_images)
    print ("Number of Negative image per Query image: "+args.num_neg_images)

    triplet_sampler(directory_path=args.input_directory, output_path=args.output_directory, num_neg_images=args.num_neg_images, num_pos_images=args.num_pos_images)
