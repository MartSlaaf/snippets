import skimage
import numpy as np
import os
from glob import glob as glob


def get_sigmed_outliers_iterative(v, alpha=3, max_i=5, window=None):
    o = np.zeros_like(v)
    small_window = max(10, int(window/10))

    corrected = deepcopy(v)
    for i in range(max_i):
        # gain median filtered values
        filtered = pd.Series(corrected).rolling(window, center=True).median()
        filtered_small = pd.Series(corrected).rolling(small_window, center=True).median()
        filtered[filtered.isna()] = filtered_small[filtered.isna()]
        filtered = filtered.fillna(filtered.mean())
        # substract median filtered values
        centered = corrected - filtered
        # detect and accumulate sigma-outliers
        s = centered.std()
        current_outliers = np.logical_or(centered<-alpha*s, centered>+alpha*s).astype(int)
        current_outliers[:small_window] = 0
        current_outliers[-small_window:] = 0
        o = np.stack([current_outliers, o]).max(0)
        # replace sigma-outliers with filtered values
        corrected[current_outliers==1] = filtered[current_outliers==1]

    return o, corrected


def correct_stack_outliers(images):
    curr_means = images.reshape(images.shape[0], -1).mean(-1)
    curr_stds = images.reshape(images.shape[0], -1).std(-1)
    o_means, corr_means = get_sigmed_outliers_iterative(curr_means, 3, window=80, max_i=30)
    o_stds, corr_stds = get_sigmed_outliers_iterative(curr_stds, 3, window=80, max_i=30)
    o_max = np.stack([o_means, o_stds]).max(0)

    c_m = (o_means==1)
    c_s = (o_stds==1)
    c_a = (o_max==1)

    # turn all means to zero
    images[c_a] -= curr_means[c_a, None, None]
    # correct selected standard deviations
    images[c_s] /= curr_stds[c_s, None, None]
    images[c_s] *= corr_stds[c_s, None, None]
    # return means to the corrected values
    images[c_a] += corr_means[c_a, None, None]

    return images


def load_images_folder(images_path, useful_ids=None, part_to_load=None, reshape_to=None,
                       rescale_intensity=None, normalize_intensities_overall=False, fit_outliers=False,
                       slicing_axis=0, circle_crop=None,
                       progressbar=None):
    """
    Loading pack of images from the folder.
    Args:
        images_path (str): path to the folder with the images. All files in the folder should be images with names to be alphabetically sorted
        useful_ids (list): list of integers to load specific images only. Should be indices in the list of sorted images
        part_to_load (float, int): partition (if float) or count (if int) of images to load. Loads random
        reshape_to (tuple): (int, int) size of picture to be at the end. Reshape happens after all other operations
        rescale_intensity (dict): dictionary of parameters to be passed to skimage.exposure.rescale_intensity
        normalize_intensities_overall (bool): if True, will shift and scale each picture based on overall statistics of loaded pictures
        slicing_axis (int): could be used to reslice the picture not on zero-axis, as it stored, but on the 1 or 2
        circle_crop (dictionary): should contain:
            key 'center': position of center of content in parts of picture (tuple of floats 0-1)
            key 'radius': float, portion of picture held by object of interest
    Returns:
        images (list): list of numpy images loaded and prepared
    """

    # preparing selected indices
    if (part_to_load is not None) and (useful_ids is None):
        if isinstance(part_to_load, float):
            part_to_load = int(part_to_load * len(all_images))
        if isinstance(part_to_load, int):
            useful_ids = np.random.randint(0, len(all_images), part_to_load)

    if os.path.isfile(images_path):
        # single file mode
        if images_path[-4:] == '.npy':
            images = np.load(images_path)
        elif images_path[-4:] == '.npz':
            images = list(np.load(images_path).values())[0]
        else:
            raise Exception('Now only supported single-file types are [.npy and .npz]')
        if (slicing_axis == 0) and (useful_ids is not None):
            images = images[useful_ids]
    else:
        # Select the right pictures to load
        all_images = sorted(glob(os.path.join(images_path , '*')))
        if (useful_ids is not None) and (slicing_axis == 0):
            all_images = [all_images[id] for id in useful_ids]

        # load images and rescale intensities if needed
        images = []
        if progressbar is not None:
            all_images = progressbar(all_images)
        for image in all_images:
            if image[-4:] == '.npy':
                img = np.load(image)
            elif image[-4:] == '.npz':
                img = list(np.load(image).values())[0]
            else:
                img = skimage.io.imread(image)
            if rescale_intensity is not None:
                img = skimage.exposure.rescale_intensity(img, **rescale_intensity)
            images.append(img)

            all_shapes = np.stack([img.shape for img in images])
            if (all_shapes.max(0) != all_shapes.min(0)).any():
                minimal_shape = all_shapes.min(0)
                cropped_images = []
                for image in images:
                    if (image.shape != minimal_shape).any():
                        cropped_images.append(crop_center(image, *minimal_shape))
                    else:
                        cropped_images.append(image)
                images = cropped_images
        images = np.stack(images)

    # swap slicing axis
    if slicing_axis != 0:
        images = np.swapaxes(images, 0, slicing_axis)
        if useful_ids is not None:
            images = images[useful_ids]

    # return outliers to the neighbours level
    if fit_outliers:
        images = correct_stack_outliers(images)

    # normalize overall intensities
    if normalize_intensities_overall:
        filtered_images = images[np.logical_and(images > np.percentile(images, 5), images < np.percentile(images, 95))]
        overall_mean, overall_var = filtered_images.mean(), filtered_images.std()
        images = (images - overall_mean)/overall_var

    # crop circle of interest
    if circle_crop is not None:
        if isinstance(circle_crop['center'][0], int):
            center_x = circle_crop['center'][0]
        else:
            center_x = circle_crop['center'][0] * images.shape[1]
        if isinstance(circle_crop['center'][1], int):
            center_y = circle_crop['center'][1]
        else:
            center_y = circle_crop['center'][1] * images.shape[2]
        radius = circle_crop['radius'] * min(images.shape[1:]) // 2

        left_x = int(center_x - radius)
        right_x = int(center_x + radius)
        left_y = int(center_y - radius)
        right_y = int(center_y + radius)

        padding_blocks = [[0, 0], [0, 0]]
        need_to_pad = False

        if left_x < 0:
            padding_blocks[0][0] = -left_x
            left_x = 0
            need_to_pad = True
        if right_x > images.shape[1] - 1:
            padding_blocks[0][1] = right_x - images.shape[1]
            right_x = images.shape[1]
            need_to_pad = True
        if left_y < 0:
            padding_blocks[1][0] = -left_y
            left_y = 0
            need_to_pad = True
        if right_y > images.shape[2] - 1:
            padding_blocks[1][1] = right_y - images.shape[2]
            need_to_pad = True


        images = images[:, left_x:right_x, left_y:right_y]

        if need_to_pad:
            new_images = np.zeros((images.shape[0], images.shape[1] + sum(padding_blocks[0]), images.shape[2] + sum(padding_blocks[1])))
            new_images[:, padding_blocks[0][0]:images.shape[1]+padding_blocks[0][0], padding_blocks[1][0]:images.shape[2]+padding_blocks[1][0]] = images
            images = new_images

    # finally reshape
    if reshape_to is not None:
        images = skimage.transform.resize(images, [images.shape[0], *reshape_to], preserve_range=True)
    return images
