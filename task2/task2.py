import numpy as np
import matplotlib.pyplot as plt
from sympy import Implies, re
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    # Compute intersection
    Left_intersection = max(prediction_box[0],gt_box[0])
    Right_intersection = min(prediction_box[2],gt_box[2])
    top_intersection =min(prediction_box[3],gt_box[3])
    bottom_intersection=max(prediction_box[1],gt_box[1])

    if Left_intersection > Right_intersection or bottom_intersection > top_intersection:
        return 0
    
    if prediction_box[0]> gt_box[2]:
        iou=0
        return iou
    if gt_box[0]> prediction_box[2]:
        iou=0
        return iou
    if prediction_box[1]> gt_box[3]:
        iou=0
        return iou
    if gt_box[1]> prediction_box[3]:
        iou=0
        return iou
        
    Intersection_area = (top_intersection-bottom_intersection)*(Right_intersection -Left_intersection)
    # Compute union
    prediction_box_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    
    Union_area = prediction_box_area + gt_area 
    iou = Intersection_area / (Union_area- Intersection_area) 
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1
    return num_tp/(num_tp+ num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0
    
    return num_tp/(num_tp+num_fn)



def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    iou_list= []
    
    
    for i in range(prediction_boxes.shape[0]):
        for j in range(gt_boxes.shape[0]):
            iou = calculate_iou(prediction_boxes[i,:], gt_boxes[j,:])
            if iou >= iou_threshold:
                iou_list.append([j,i,iou])

    # Sort all matches on IoU in descending order
    highst_prediction_boxes_matched = []
    highst_gt_boxes_matched = []

    iou_list.sort(key=lambda x: x[2])
    iou_list.sort(key=lambda x: x[0])
    
    # Find all matches with the highest IoU threshold
    previous_gtid = gt_boxes.shape[0]+1
    for k in range(len(iou_list)):
        actual_gtid = iou_list[k][0]
        if actual_gtid is not previous_gtid:
            highst_prediction_boxes_matched.append(prediction_boxes[iou_list[k][1]])
            highst_gt_boxes_matched.append(gt_boxes[iou_list[k][0]])
            previous_gtid =actual_gtid

    return np.array(highst_prediction_boxes_matched), np.array(highst_gt_boxes_matched)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    prediction_boxes_matched, gt_boxes_matched = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    
    num_tp = prediction_boxes_matched.shape[0]
    num_fp = prediction_boxes.shape[0] - num_tp
    num_fn = gt_boxes.shape[0] - num_tp
    
    return {"true_pos": num_tp, "false_pos": num_fp, "false_neg": num_fn}
    
    
    

def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    
    num_tp = 0
    num_fp = 0
    num_fn = 0
    
    for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        count = calculate_individual_image_result (prediction_boxes, gt_boxes, iou_threshold)
        num_tp = num_tp + count["true_pos"]
        num_fp = num_fp + count["false_pos"]
        num_fn = num_fn + count["false_neg"]
    
    Precision = calculate_precision(num_tp, num_fp, num_fn)
    Recall = calculate_recall(num_tp, num_fp, num_fn)
    return Precision, Recall


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []
    
    
    for confidence_treshold in confidence_thresholds:
        prediction_box_image = []
        for i, j in enumerate(all_prediction_boxes):
            prediction_b = []
            for k in range(len(j)):
                if confidence_scores[i][k] >= confidence_treshold:
                    prediction_b.append(all_prediction_boxes[i][k])
            prediction_box_image. append(np.array(prediction_b))
            
        Precision, Recall = calculate_precision_recall_all_images(prediction_box_image,all_gt_boxes, iou_threshold)
            
        precisions.append(Precision)
        recalls.append(Recall)
            
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    AP = 0
    i = 0
    for recall_level in recall_levels:
        max_precision = 0
        for j in range(precisions.shape[0]):
            if precisions[j] >= max_precision and recalls[j] >= recall_level:
                max_precision = precisions[j]
        AP = AP + max_precision
        i =i+1
    mAP = AP/i
    return mAP
        
        
        
        
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
