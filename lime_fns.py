import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.segmentation import mark_boundaries
from lime import lime_image
from utils_fns import load_img
import numpy as np
import sam_fns


def get_explanation_w_sam(image_path, model, mask_generator, img_size=(250, 250)):
    '''takes an img and the model used for the prediction\n
    returns the img explained (img + mask)'''

    img = load_img(image_path, img_size)
    img_for_prediction = tf.expand_dims(img/255.0, axis=0) #resizing and bathcing img
    img_for_explanation = np.clip(img.numpy(), 0, 255).astype(np.uint8)

    #instancing explainer
    explainer = lime_image.LimeImageExplainer()

    #generating explanation
    explanation = explainer.explain_instance(
    img_for_explanation.astype('double'),
    model.predict,
    top_labels=2,
    num_features=20,
    hide_color=0,
    num_samples=2000,
    segmentation_fn=lambda img: sam_fns.sam_segmentation_fn(img, mask_generator),
    segmentation_fn_img = None)

    #viewing explanation
    temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=False,
    num_features=5,
    hide_rest=False)

    explained_img = mark_boundaries(temp / 255, mask)

    return explained_img

def misclassifications_explanation_w_sam(img_paths, class_labels, true_labels, pred_labels, model):
    '''returns a list with tuples (img_with_explanation_mask, true_label, pred_label)'''
    true_labels = np.array(true_labels)
    class_labels = np.array(class_labels)
    pred_labels = class_labels[pred_labels]
    
    errors = [(path, true, pred) for path, true, pred in zip(img_paths, true_labels, pred_labels) if true != pred]

    imgs_explained = [(get_explanation_w_sam(img_path, model), true_label, pred_label) for img_path, true_label, pred_label in errors]

    return imgs_explained

'''def misclassifications_explanation(img_paths, class_labels, true_labels, pred_prob, model):
    #returns a list with tuples (img_with_explanation_mask, true_label, pred_label)
    true_labels = np.array(true_labels)
    class_labels = np.array(class_labels)
    print(class_labels)
    pred_prob = np.argmax(pred_prob, axis=1)
    pred_labels = class_labels[pred_prob]
    
    errors = [(path, true, pred) for path, true, pred in zip(img_paths, true_labels, pred_labels) if true != pred]

    imgs_explained = [(get_explanation(img_path, model), true_label, pred_label) for img_path, true_label, pred_label in errors]

    return imgs_explained
'''
def visualize_explanations(imgs_explained):
    '''plots explained imgs with its corresponding true and pred labels'''
    plt.figure(figsize=(10, 10))

    for i, (img, true_label, pred_label) in enumerate(imgs_explained):
        plt.subplot(5, 3, i + 1)  #defining img's position on the grid
        plt.imshow(img)
        plt.title(f'True: {true_label}, Pred: {pred_label}')
        plt.axis("off")

    plt.tight_layout()
    plt.show()