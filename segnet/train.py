#%%

from keras_segmentation.models.segnet import resnet50_segnet

# def custom_augmentation():
#     return  iaa.Sequential(
#         [
#             # apply the following augmenters to most images
#             iaa.Fliplr(0.5),  # horizontally flip 50% of all images
#             iaa.Flipud(0.5), # horizontally flip 50% of all images
#         ])


def train (model, n_classes, train_images, train_annotations, val_images, val_annotations,
                                            name_dataset, name_model, batch_size = 2, epochs = 5):
    model.train(
        train_images =  train_images,
        train_annotations = train_annotations,

        val_images = val_images,
        val_annotations = val_annotations,

        batch_size = batch_size,
        val_batch_size = batch_size,

        steps_per_epoch= 20210 // batch_size,
        val_steps_per_epoch= 2000 // batch_size,

        checkpoints_path =  'tmp/' + name_dataset + '/' + name_model + '/', 
        epochs= epochs,

        # do_augment=True, # enable augmentation 
        # custom_augmentation=custom_augmentation # sets the augmention function to use
    )

    model.save_weights("pretrained/" + name_dataset + '/' + name_model + '_' + str(n_classes) + '/' )

#%%

n_classes = 200
model = resnet50_segnet(n_classes=n_classes ,  input_height=416, input_width=608 ) 

# train_images = '../datasets/ADEChallengeData2016/images/training/'
# train_annotations = '../datasets/ADEChallengeData2016/annotations/training/'
# val_images = '../datasets/ADEChallengeData2016/images/validation/'
# val_annotations = '../datasets/ADEChallengeData2016/images/validation/'

# train_images = '../datasets/dataset1/images_prepped_train/'
# train_annotations = '../datasets/dataset1/annotations_prepped_train/'
# val_images = None
# val_annotations = None

name_dataset = 'dataset1'
name_model = 'resnet50_segnet'
batch_size = 10
epochs = 10

#%%

train (model, n_classes, train_images, train_annotations, val_images, val_annotations,
                                            name_dataset, name_model, batch_size, epochs)