import os
import autoaugment
import tensorflow as tf

def Augmentation(read_path, save_path, i):

    img = tf.io.read_file(read_path)
    image = tf.image.decode_jpeg(img, channels=3)
    aug_image = autoaugment.distort_image_with_autoaugment(image, 'v0')

    img = tf.image.encode_jpeg(aug_image)
    img_save_path = save_path.split('.jpg')[0] + '_' + str(i) + '.jpg'
    with tf.io.gfile.GFile(img_save_path, 'wb') as file:
        file.write(img.numpy())


imgs_path_m = 'dataset/train_images/Monkey Pox/'
imgs_path_o = 'dataset/train_images/Others/'
save_path_m = 'dataset/augmented_train_images/Monkey Pox/'
save_path_o = 'dataset/augmented_train_images/others/'


for filename in os.listdir(imgs_path_m):
    img_path = imgs_path_m + filename
    save_path = save_path_m + filename
    for i in range(15):
        Augmentation(img_path, save_path, i)

for filename in os.listdir(imgs_path_o):
    img_path = imgs_path_o + filename
    save_path = save_path_o + filename
    for i in range(15):
        Augmentation(img_path, save_path, i)
