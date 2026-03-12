def fuse_predictions(text_label, text_score, image_label, image_score):

    try:
        text_score = float(text_score)
    except:
        text_score = 0.5

    try:
        image_score = float(image_score)
    except:
        image_score = 0.5

    # Case 1: Both models agree
    if text_label == image_label:

        final_label = text_label
        final_score = (text_score + image_score) / 2

    # Case 2: Models disagree
    else:

        weighted_text = 0.6 * text_score
        weighted_image = 0.4 * image_score

        if weighted_text > weighted_image:
            final_label = text_label
            final_score = weighted_text
        else:
            final_label = image_label
            final_score = weighted_image

    return final_label, final_score