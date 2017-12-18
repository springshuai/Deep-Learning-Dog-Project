def extract_VGG16(tensor):
	from keras.applications.vgg16 import VGG16, preprocess_input
	return VGG16(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_VGG19(tensor):
	from keras.applications.vgg19 import VGG19, preprocess_input
	return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Resnet50(tensor):
	from keras.applications.resnet50 import ResNet50, preprocess_input
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Xception(tensor):
	from keras.applications.xception import Xception, preprocess_input
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_InceptionV3(tensor):
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# SHUAI added

##def path_to_tensor_plus_prep(img_path):
##    from keras.preprocessing import image  
##    # loads RGB image as PIL.Image.Image type
##    img = image.load_img(img_path, target_size=(224, 224))
##    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
##    x = image.img_to_array(img)
##    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor       
##    return densenet.preprocess_input(np.expand_dims(x, axis=0))

def extract_DenseNet(path):
    import densenet
    from keras.preprocessing import image
    import numpy as np
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    img_tensor =  densenet.preprocess_input(np.expand_dims(x, axis=0))
    image_dim = (224, 224, 3)
    DenseNet_model = densenet.DenseNetImageNet161(input_shape=image_dim, include_top=False)    
    DenseNet_output = DenseNet_model.predict(img_tensor)
    return DenseNet_output
