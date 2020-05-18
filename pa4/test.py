from PIL import Image
from convertToGrayscale import ConvertToGrayscale
import numpy
from random import randint
from copy import deepcopy
from scipy import spatial
from collections import Counter

MAX_ITERATIONS = 1500


class Test(ConvertToGrayscale):
    def __init__(self, k, img_base_dir, img_file_name, image_dir, createGrayscaleRegardless, AGENT):
        super().__init__(img_base_dir, img_file_name, image_dir, createGrayscaleRegardless, AGENT)
        self.k = k
        self.grayscaleImageConversion()

        self.np_img_values = numpy.array(deepcopy(self.img_RGB_values))
        self.grayscale_image = Image.open(self.grayscale_img_file_path)
        self.img_grayscale_RGB_values = list(self.grayscale_image.getdata())

        self.right_side_patches = None
        self.left_side_patches = None
        self.right_side_patch_map = {}
        self.left_side_patch_map = {}
        self.centroids = None
        print("Image Location: ", self.original_img_file_path)
        print("Image Dimensions: ", self.width, "x", self.height)

    def isPixelInImageBounds(self, coordinate):
        (x, y) = coordinate
        return (0 <= x < self.width) and (0 <= y < self.height)

    def filter_right_overlap(self, coordinate):
        (x, y) = coordinate
        return (int(self.width / 2) <= x < self.width) and (0 <= y < self.height)

    def filter_left_overlap(self, coordinate):
        (x, y) = coordinate
        return (0 <= x < int(self.width / 2)) and (0 <= y < self.height)

    def get_pixel_neighbors(self, coordinate):
        (x, y) = coordinate  #
        neighboring_pixels = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y),
                              (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]
        return list(filter(self.isPixelInImageBounds, neighboring_pixels))

    def preprocess_grayscale_sides(self):
        print("Pre-processing Grayscale")
        left_side_patches = []
        right_side_patches = []
        iteration_left, iteration_right = 0, 0
        for y in range(self.height):
            for x in range(self.width):

                pixel_coordinate = (x, y)
                pixel_neighbors = self.get_pixel_neighbors(pixel_coordinate)
                if x < int(self.width / 2):
                    pixel_neighbors = list(filter(self.filter_left_overlap, pixel_neighbors))
                else:
                    pixel_neighbors = list(filter(self.filter_right_overlap, pixel_neighbors))
                if len(pixel_neighbors) == 0 or len(pixel_neighbors) < 9:
                    continue
                pixel_patch = []
                for pixel_neighbor in pixel_neighbors:
                    (x_p, y_p) = pixel_neighbor
                    pixel_patch_element = list(self.img_RGB_values[x_p + (self.width * y_p)])
                    pixel_patch.append(pixel_patch_element)
                pixel_patch_np = numpy.array(pixel_patch)
                # print("P: ", pixel_patch_np)

                pixel_patch_ = numpy.mean(pixel_patch_np, axis=0, dtype=numpy.int32)
                if x < int(self.width / 2):
                    left_side_patches.append(numpy.array(pixel_patch_))
                    self.left_side_patch_map[iteration_left] = (x, y)
                    iteration_left = iteration_left + 1
                else:
                    right_side_patches.append(numpy.array(pixel_patch_))
                    self.right_side_patch_map[iteration_right] = (x, y)
                    iteration_right = iteration_right + 1
        # right_side_patches_ = numpy.array(right_side_patches).T
        self.right_side_patches = numpy.array(right_side_patches)
        self.left_side_patches = numpy.array(left_side_patches)
        print("Pre-processing Complete")

    def representativePixelColor(self, isPixelInCluster):
        temp_list = [isPixelInCluster]
        pixel_np = numpy.array(temp_list)
        pixel_distances = spatial.distance.cdist(pixel_np, self.centroids)
        minimum_pixel_distances = numpy.argmin(pixel_distances, axis=1)
        # print(minimum_pixel_distances, " | R: ", self.centroids[minimum_pixel_distances[0]])
        representative_colored_pixel = tuple(self.centroids[minimum_pixel_distances[0]])
        return representative_colored_pixel

    def computeSimilarPatches(self):
        print("Finding Similar Patches...")
        self.mindist = spatial.distance.cdist(self.right_side_patches, self.centroids)
        self.test_min = numpy.argmin(self.mindist, axis=1)
        labels = {tuple(c): [] for c in self.centroids}
        index = 0
        for pixel in self.right_side_patches:
            try:
                centroid_key = tuple(self.centroids[self.test_min[index]])

                labels[centroid_key].append(pixel)

                index = index + 1
            except IndexError:
                print(pixel, " Err: ", index)
                print(labels)
                break
        print("Search Complete")
        return labels

    def representativePixelColor_(self, isPixelInCluster):
        temp_list = [isPixelInCluster]
        pixel_np = numpy.array(temp_list)
        pixel_distances = spatial.distance.cdist(pixel_np, self.centroids)
        minimum_pixel_distances = numpy.argmin(pixel_distances, axis=1)
        # print(minimum_pixel_distances, " | R: ", self.centroids[minimum_pixel_distances[0]])
        representative_colored_pixel = tuple(self.centroids[minimum_pixel_distances[0]])
        return representative_colored_pixel

    def recolor_right_side(self):
        print("Recoloring Right Side Basic Algo....")
        for index in range(len(self.right_side_patches)):
            (x_r, y_r) = self.right_side_patch_map[index]
            pixel_ = self.img_grayscale_RGB_values[x_r + (self.width * y_r)]
            color = self.representativePixelColor_(pixel_)
            self.image_edit[x_r, y_r] = color
        self.original_image.save(self.recolor_right_img_file_path)
        print("Recoloring Complete....")

    def preprocess_grayscale_sides_kd(self):
        print("Pre-processing Grayscale")
        left_side_patches = []
        right_side_patches = []
        patch_index_left, patch_index_right = 0, 0
        for y in range(self.height):
            for x in range(self.width):
                pixel_coordinate = (x, y)
                pixel_neighbors = self.get_pixel_neighbors(pixel_coordinate)

                if x < int(self.width / 2):
                    pixel_neighbors = list(filter(self.filter_left_overlap, pixel_neighbors))
                else:
                    pixel_neighbors = list(filter(self.filter_right_overlap, pixel_neighbors))

                if len(pixel_neighbors) == 0 or len(pixel_neighbors) < 9:
                    # print("BAD: (",x,",",y_,")")
                    continue

                pixel_patch = []
                for pixel_neighbor in pixel_neighbors:
                    (x_p, y_p) = pixel_neighbor
                    pixel_patch_element = [(self.img_grayscale_RGB_values[x_p + (self.width * y_p)])[0]]
                    pixel_patch.append(pixel_patch_element)
                pixel_patch_np = numpy.array(pixel_patch)
                pixel_patch_ = pixel_patch_np.flatten()
                # print("P: ", pixel_patch_np)
                patch_index = x + (self.width * y)

                if x < int(self.width / 2):
                    left_side_patches.append(pixel_patch_)
                    self.left_side_patch_map[patch_index_left] = (x, y)
                    patch_index_left = patch_index_left + 1
                else:
                    right_side_patches.append(pixel_patch_)
                    self.right_side_patch_map[patch_index_right] = (x, y)
                    patch_index_right = patch_index_right + 1

        self.right_side_patches = right_side_patches
        self.left_side_patches = numpy.array(left_side_patches)
        print("Pre-processing Complete")

    def selectColorForRightSide_kd(self, index_of_patches):
        potential_middle_pixel_colors = []
        pixel_color_mapping = {}
        pixel_coordinate_map = {}
        for index in index_of_patches:
            pixel_left_side_coordinate = self.left_side_patch_map[index]
            (x_l, y_l) = pixel_left_side_coordinate
            representative_color = self.img_RGB_values[x_l + (self.width * y_l)]
            potential_middle_pixel_colors.append(representative_color)

            pixel_color_mapping[index] = representative_color
            pixel_coordinate_map[index] = pixel_left_side_coordinate

        frequency_of_rgb = Counter(potential_middle_pixel_colors)
        # print(index_of_patches)
        # print("M: ", pixel_color_mapping)
        # print("S: ", pixel_coordinate_map)

        # print(frequency_of_rgb)
        max_frequency_in_RGB = max(v for _, v in frequency_of_rgb.items())
        list_of_coordinates = [key for key, value in frequency_of_rgb.items() if value == max_frequency_in_RGB]
        # print("List of coord: ", list_of_coordinates)

        pixel_coordinate_to_recolor = None
        if len(list_of_coordinates) >= 1:  # break ties in order
            for id in index_of_patches:
                if pixel_color_mapping[id] not in list_of_coordinates:
                    continue
                else:
                    pixel_coordinate_to_recolor = pixel_coordinate_map[id]
                    break
        else:  # 0 index is most similar color, so if no ties can be broken default to most similar
            pixel_coordinate_to_recolor = self.left_side_patch_map[index_of_patches[0]]

        (x_l, y_l) = pixel_coordinate_to_recolor
        representative_color_ = self.img_RGB_values[x_l + (self.width * y_l)]
        # print("Recolor: ", pixel_coordinate_to_recolor, " w/ ", representative_color_)
        return representative_color_

    def getIDsOfClosestPatches(self, all_minimum_distances):
        k_minimum_ids = list(numpy.argpartition(all_minimum_distances, 6))
        subset_k_minimum_ids = k_minimum_ids[:6]
        minimum_distance = list(all_minimum_distances[subset_k_minimum_ids])

        zipped_minimum_params = zip(minimum_distance, subset_k_minimum_ids)
        sorted_minimum_params = sorted(zipped_minimum_params)
        minimum_params_association = zip(*sorted_minimum_params)

        sorted_ids_by_minimum_distances = [list(elem) for elem in minimum_params_association]
        return sorted_ids_by_minimum_distances

    def recolor_right_side_kd(self, iteration, color):
        (x_r, y_r) = self.right_side_patch_map[iteration]
        self.image_edit[x_r, y_r] = color

    def computeSimilarPatches_kd(self):
        print("Coloring Right Side Patches...")
        iteration = 0
        for right_side_patch in self.right_side_patches:
            # print("Patch: ", right_side_patch)
            temp_r_patch = [right_side_patch]
            temp_r_patch_np = numpy.array(temp_r_patch)

            tree = spatial.cKDTree(temp_r_patch_np)
            all_minimum_distances, temp = tree.query(self.left_side_patches)
            min_dist, min_id = self.getIDsOfClosestPatches(all_minimum_distances)
            color = self.selectColorForRightSide_kd(min_id)
            self.recolor_right_side_kd(iteration, color)
            iteration = iteration + 1
        self.original_image.save(self.recolor_right_img_file_path)
        print("Right Side Coloring Complete...")

    def compute(self):
        print("Computing Clusters....")
        centroids = self.generateRandomCentroids(self.k + 1)
        iterations = 0
        oldCentroids = None
        while not self.isClusteringComplete(oldCentroids, centroids, iterations):
            # if not oldCentroids is None:
            # print("Old Centroids: ", oldCentroids, "New Centroids: ", centroids)
            oldCentroids = centroids
            iterations = iterations + 1
            labels = self.getLabels(centroids)
            centroids = self.getCentroids(labels)
        print("Compute Complete")
        self.centroids = centroids
        return centroids

    def isClusteringComplete(self, oldCentroids, centroids, iterations):
        if iterations > MAX_ITERATIONS: return True
        return numpy.array_equal(oldCentroids, centroids)

    def generateRandomCentroids(self, k):
        print("Generating Random Centroids...")

        if self.width % 2 == 0:
            training_data_width = int(((self.width) / 2))
        else:
            training_data_width = int(((self.width - 1) / 2))

        used_centroids, centroids = [], []
        while len(used_centroids) < k:
            random_x = randint(0, training_data_width)
            random_y = randint(0, self.height)

            potential_centroid = self.img_RGB_values[random_x + (self.width * random_y)]
            if list(potential_centroid) not in used_centroids:
                used_centroids.append(list(potential_centroid))
                centroids.append(potential_centroid)
        print("Generation Complete")
        return numpy.array(centroids)

    def getCentroids(self, labels):
        print("Updating Clustering Labels...")
        updated_labels = []
        for centroid, clustered_points in labels.items():
            new_c = numpy.mean(clustered_points, axis=0, dtype=numpy.int32)
            updated_labels.append(numpy.array(new_c))
        print("Update Complete")
        return numpy.array(updated_labels)

    def getLabels(self, centroids):
        print("Finding Nearest Centroid...")
        self.mindist = spatial.distance.cdist(self.np_img_values, centroids)
        self.test_min = numpy.argmin(self.mindist, axis=1)
        labels = {tuple(c): [] for c in centroids}
        index = 0
        for pixel in self.np_img_values:
            try:
                centroid_key = tuple(centroids[self.test_min[index]])
                labels[centroid_key].append(pixel)
                index = index + 1
            except IndexError:
                print(pixel, " Err: ", index)
                print(labels)
                break
        print("Search Complete")
        return labels
