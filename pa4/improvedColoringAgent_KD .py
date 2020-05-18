from sys import maxsize
from convertToGrayscale import ConvertToGrayscale
from random import randint
from copy import deepcopy
from scipy import spatial
import numpy
from PIL import Image
from definitions import DEFINITIONS
from collections import Counter


class ImprovedColoringAgentKD(ConvertToGrayscale):
    def __init__(self, k, img_base_dir, img_file_name, image_dir,
                 createGrayscaleRegardless, disableProgramConsoleLog, AGENT):
        super().__init__(img_base_dir, img_file_name, image_dir, createGrayscaleRegardless, AGENT)
        print("Running Improved KD Agent Using KMeans++...")
        self.k = k
        self.disableProgramConsoleLog = disableProgramConsoleLog
        self.grayscaleImageConversion()

        # Setup Image File Path Variables
        self.original_image.save(self.grayscale_img_file_path)
        self.np_img_values = numpy.array(deepcopy(self.img_RGB_values))

        self.grayscale_image = Image.open(self.grayscale_img_file_path)
        self.img_grayscale_RGB_values = list(self.grayscale_image.getdata())

        # Clustering
        self.clusters = None
        self.centroids = None
        self.clustered_list = {}
        self.disableProgramConsoleLog = disableProgramConsoleLog

        # Right Side Prediction Variables
        self.left_side_patches = None
        self.left_side_patch_map = {}

        self.right_side_patches = None
        self.right_side_patch_map = {}

        self.predicted_img_RGB_values = None


    def getEuclideanDist(self, point1, point2):
        return numpy.sum((point1-point2)**2)

    def kMeansPPInit(self):
        print("Generating Centroids Using K-Means++...")
        if self.width % 2 == 0:
            training_data_width = int(((self.width) / 2))
        else:
            training_data_width = int(((self.width - 1) / 2))
        used_centroids, centroids = [], []
        random_x = randint(0, training_data_width)
        random_y = randint(0, self.height - 1)
        firstRandomCentroid = self.img_RGB_values[random_x + (self.width * random_y)]
        centroids.append(firstRandomCentroid)
        data = self.np_img_values
        for x in range(self.k-1):
            distList = []
            for i in range(data.shape[0]):
                point = data[i, :]
                d = maxsize

                for j in range(len(centroids)):
                    temp= self.getEuclideanDist(point, centroids[j])
                    #print("got distance")
                    d = min(d, temp)
                distList.append(d)

            dList = numpy.array(distList)
            nextCentroid = data[numpy.argmax(dList), :]
            centroids.append(nextCentroid)
        print("K-Means++ Compute Complete")
        return numpy.array(centroids)


    def compute_clusters(self):
        print("Computing Clusters....")
        centroids = self.kMeansPPInit()
        iterations = 0
        oldCentroids = None
        while not self.isClusteringComplete(oldCentroids, centroids, iterations):
            # if not oldCentroids is None:
            # print("Old Centroids: ", oldCentroids, "New Centroids: ", centroids)
            oldCentroids = centroids
            iterations = iterations + 1
            clusters = self.cluster_pixels(centroids)
            centroids = self.recomputeCentroids(clusters)
        self.clusters = self.cluster_pixels(centroids)
        self.centroids = centroids
        print("Compute Complete")
        return centroids

    def isClusteringComplete(self, oldCentroids, centroids, iterations):
        if iterations > DEFINITIONS.MAX_ITERATIONS: return True
        return numpy.array_equal(oldCentroids, centroids)

    def generateRandomCentroids(self):
        print("Generating Random Centroids...")
        if self.width % 2 == 0:
            training_data_width = int(((self.width) / 2))
        else:
            training_data_width = int(((self.width - 1) / 2))

        used_centroids, centroids = [], []
        while len(used_centroids) < self.k:
            random_x = randint(0, training_data_width)
            random_y = randint(0, self.height - 1)
            potential_centroid = self.img_RGB_values[random_x + (self.width * random_y)]
            if list(potential_centroid) not in used_centroids:
                used_centroids.append(list(potential_centroid))
                centroids.append(potential_centroid)
        print("Generation Complete")
        return numpy.array(centroids)

    def recomputeCentroids(self, labels):
        if not self.disableProgramConsoleLog == "consoleLog-centroid-calculations":
            print("Recalculating Clustering Labels...")
        updated_centroids = []
        for centroid, clustered_points in labels.items():
            new_c = numpy.mean(clustered_points, axis=0, dtype=numpy.int32)
            updated_centroids.append(numpy.array(new_c))
        if not self.disableProgramConsoleLog == "consoleLog-centroid-calculations":
            print("Recalculation Complete")
        return numpy.array(updated_centroids)

    def cluster_pixels(self, centroids):
        if not self.disableProgramConsoleLog == "consoleLog-centroid-calculations":
            print("Clustering Pixels To Centroid...")
        pixel_distances = spatial.distance.cdist(self.np_img_values, centroids)
        minimum_pixel_distances = numpy.argmin(pixel_distances, axis=1)
        labels = {tuple(c): [] for c in centroids}
        index = 0
        for pixel in self.np_img_values:
            centroid_key = tuple(centroids[minimum_pixel_distances[index]])
            labels[centroid_key].append(pixel)
            index = index + 1
        if not self.disableProgramConsoleLog == "consoleLog-centroid-calculations":
            print("Clustering Complete")
        return labels

    def representativePixelColor(self, isPixelInCluster):
        temp_list = [isPixelInCluster]
        pixel_np = numpy.array(temp_list)
        pixel_distances = spatial.distance.cdist(pixel_np, self.centroids)
        minimum_pixel_distances = numpy.argmin(pixel_distances, axis=1)
        # print(minimum_pixel_distances, " | R: ", self.centroids[minimum_pixel_distances[0]])
        representative_colored_pixel = tuple(self.centroids[minimum_pixel_distances[0]])
        return representative_colored_pixel

    def recolor_left_side_of_image(self):
        print("Recoloring Left Side Of Image...")

        for y in range(self.height):
            for x in range(0, int(self.width / 2)):
                pixel = self.img_RGB_values[x + (self.width * y)]
                self.image_edit[x, y] = self.representativePixelColor(pixel)
        print("Left Side Recoloring Complete")
        self.original_image.save(self.recolor_left_img_file_path)

    # RIGHT RECOLOR Functions

    def isPixelInImageBounds(self, coordinate):
        (x, y) = coordinate
        return (0 <= x < self.width) and (0 <= y < self.height)

    def get_pixel_neighbors(self, coordinate):
        (x, y) = coordinate  #
        neighboring_pixels = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y),
                              (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]
        return list(filter(self.isPixelInImageBounds, neighboring_pixels))

    def filter_right_overlap(self, coordinate):
        (x, y) = coordinate
        return (int(self.width / 2) <= x < self.width) and (0 <= y < self.height)

    def filter_left_overlap(self, coordinate):
        (x, y) = coordinate
        return (0 <= x < int(self.width / 2)) and (0 <= y < self.height)

    def preprocess_grayscale_sides(self):
        print("Pre-processing Grayscale")
        left_side_patches = []
        right_side_patches = []
        patch_index_left, patch_index_right = 0, 0
        for y in range(self.height):
            for x in range(self.width):
                pixel_coordinate = (x, y)
                pixel_neighbors = self.get_pixel_neighbors(pixel_coordinate)

                if x < int(self.width / 2):
                    pixel_n = list(filter(self.filter_left_overlap, pixel_neighbors))
                else:
                    pixel_n = list(filter(self.filter_right_overlap, pixel_neighbors))

                if len(pixel_n) == 0 or len(pixel_n) < 9:
                    continue

                pixel_patch = []
                for pixel_neighbor in pixel_n:
                    (x_p, y_p) = pixel_neighbor
                    pixel_patch_element = [(self.img_grayscale_RGB_values[x_p + (self.width * y_p)])[0]]
                    pixel_patch.append(pixel_patch_element)
                pixel_patch_np = numpy.array(pixel_patch)
                pixel_patch_ = pixel_patch_np.flatten()

                # patch_index = x + (self.width * y)

                if x < int(self.width / 2):
                    left_side_patches.append(pixel_patch_)
                    self.left_side_patch_map[patch_index_left] = (x, y)
                    patch_index_left = patch_index_left + 1
                else:
                    right_side_patches.append(pixel_patch_)
                    self.right_side_patch_map[patch_index_right] = (x, y)
                    patch_index_right = patch_index_right + 1
        # right_side_patches_ = numpy.array(right_side_patches).T
        self.right_side_patches = right_side_patches
        self.left_side_patches = numpy.array(left_side_patches)
        print("Pre-processing Complete: ", len(self.right_side_patches), " | ", len(self.left_side_patches))

    def selectColorForRightSide(self, index_of_patches):
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

        max_frequency_in_RGB = max(v for _, v in frequency_of_rgb.items())
        list_of_coordinates = [key for key, value in frequency_of_rgb.items() if value == max_frequency_in_RGB]

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
        pixel_ = self.img_RGB_values[x_l + (self.width * y_l)]
        representative_color_ = self.representativePixelColor(pixel_)
        return representative_color_

    def getIDsOfKClosestPatches(self, all_minimum_distances):
        k_minimum_ids = list(numpy.argpartition(all_minimum_distances, 6))
        subset_k_minimum_ids = k_minimum_ids[:6]
        minimum_distance = list(all_minimum_distances[subset_k_minimum_ids])

        zipped_minimum_params = zip(minimum_distance, subset_k_minimum_ids)
        sorted_minimum_params = sorted(zipped_minimum_params)
        minimum_params_association = zip(*sorted_minimum_params)

        sorted_ids_by_minimum_distances = [list(elem) for elem in minimum_params_association]

        return sorted_ids_by_minimum_distances

    def recolor_right_side(self, iteration, color):
        (x_r, y_r) = self.right_side_patch_map[iteration]
        self.image_edit[x_r, y_r] = color

    def computeSimilarPatches(self):
        print("Recoloring Right Side Patches...")
        iteration = 0
        for right_side_patch in self.right_side_patches:
            temp_r_patch = [right_side_patch]
            temp_r_patch_np = numpy.array(temp_r_patch)
            tree = spatial.cKDTree(temp_r_patch_np)
            all_minimum_distances, temp = tree.query(self.left_side_patches)
            min_dist, min_id = self.getIDsOfKClosestPatches(all_minimum_distances)
            color = self.selectColorForRightSide(min_id)
            self.recolor_right_side(iteration, color)
            if iteration % int((self.width / 2)) == 0:
                print("Recoloring...")
                self.original_image.save(self.recolor_right_img_file_path)
            iteration = iteration + 1
        self.original_image.save(self.final_recolored_img_file_path)
        print("Right Side Recoloring Complete...")

    def recolor_image(self):
        self.compute_clusters()
        self.recolor_left_side_of_image()

        self.preprocess_grayscale_sides()
        self.computeSimilarPatches()
        print("Full Image Recoloring Complete")
        final_recoloring = Image.open(self.final_recolored_img_file_path)
        #final_recoloring.show()
        self.predicted_img_RGB_values = list(final_recoloring.getdata())
        print("Improved KD Agent Complete")

    def close_image_files(self):
        self.original_image.close()
        self.grayscale_image.close()
