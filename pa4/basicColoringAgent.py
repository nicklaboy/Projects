from convertToGrayscale import ConvertToGrayscale
from random import randint
from copy import deepcopy
from scipy import spatial
import numpy
from PIL import Image
from definitions import DEFINITIONS


class BasicColoringAgent(ConvertToGrayscale):
    def __init__(self, k, img_base_dir, img_file_name, image_dir,
                 createGrayscaleRegardless, disableProgramConsoleLog, AGENT):
        super().__init__(img_base_dir, img_file_name, image_dir, createGrayscaleRegardless, AGENT)
        print("Running Basic Coloring Agent...")
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

        # Right Side Variables
        self.right_side_patches = None
        self.right_side_patch_map = {}

        self.predicted_img_RGB_values = None

    def compute_clusters(self):
        print("Computing Clusters....")
        centroids = self.generateRandomCentroids()
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
        print("Pre-Processing Right Side Grayscale Data...")
        right_side_patches = []
        iteration_right = 0
        for y in range(self.height):
            for x in range(int(self.width / 2), self.width):

                pixel_coordinate = (x, y)
                pixel_neighbors = self.get_pixel_neighbors(pixel_coordinate)

                pixel_n = list(filter(self.filter_right_overlap, pixel_neighbors))
                if len(pixel_n) == 0 or len(pixel_n) < 9:
                    continue
                pixel_patch = []
                for pixel_neighbor in pixel_n:
                    (x_p, y_p) = pixel_neighbor
                    pixel_patch_element = list(self.img_grayscale_RGB_values[x_p + (self.width * y_p)])
                    pixel_patch.append(pixel_patch_element)

                pixel_patch_np = numpy.array(pixel_patch)
                pixel_patch_ = numpy.mean(pixel_patch_np, axis=0, dtype=numpy.int32)

                right_side_patches.append(numpy.array(pixel_patch_))
                self.right_side_patch_map[iteration_right] = (x, y)
                iteration_right = iteration_right + 1

        self.right_side_patches = numpy.array(right_side_patches)
        print("Pre-Processing Complete")

    def recolor_right_side(self):
        print("Recoloring Right Side Of Image....")
        for index in range(len(self.right_side_patches)):
            (x_r, y_r) = self.right_side_patch_map[index]
            pixel_ = self.img_grayscale_RGB_values[x_r + (self.width * y_r)]
            color = self.representativePixelColor(pixel_)
            self.image_edit[x_r, y_r] = color
        # self.original_image.save(self.recolor_right_img_file_path)
        self.original_image.save(self.final_recolored_img_file_path)

        print("Right Side Recoloring Complete....")

    def recolor_image(self):
        self.compute_clusters()
        self.recolor_left_side_of_image()
        # self.k = 6
        # self.compute_clusters()
        self.preprocess_grayscale_sides()
        self.recolor_right_side()

        print("Full Image Recoloring Complete")

        final_recoloring = Image.open(self.final_recolored_img_file_path)
        # final_recoloring.show()
        self.predicted_img_RGB_values = list(final_recoloring.getdata())
        print("Basic Coloring Agent Complete")


    def close_image_files(self):
        self.original_image.close()
        self.grayscale_image.close()
