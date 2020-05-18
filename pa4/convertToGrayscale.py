from PIL import Image
import os
from definitions import DEFINITIONS

BASE_DIR = os.getcwd()

print("BASE PROJECT DIR: ", BASE_DIR)


def RGB_TO_GRAY(RGB):
    return int(0.21 * RGB[DEFINITIONS.RED] + 0.72 * RGB[DEFINITIONS.GREEN] + 0.07 * RGB[DEFINITIONS.BLUE])


class ConvertToGrayscale:
    def __init__(self, img_base_dir, img_file_name, image_dir, createGrayscaleRegardless, AGENT):

        # Setup Image DIR and File Path Variables
        self.createGrayscaleRegardless = createGrayscaleRegardless

        self.img_file_name = img_file_name
        self.TEST_FOLDER = BASE_DIR + DEFINITIONS.TEST_IMG_ASSETS_DIR
        self.IMAGE_DIR_PATH = self.TEST_FOLDER + image_dir + "/"
        self.IMAGE_RECOLOR_DIR_PATH = self.IMAGE_DIR_PATH + AGENT + "/"
        self.CREATE_RECOLORED_IMAGE_DIR()

        self.original_img_file_path = BASE_DIR + img_base_dir + img_file_name

        self.grayscale_img_file_path = self.IMAGE_DIR_PATH + "GRAYSCALE_" + img_file_name
        self.final_recolored_img_file_path = self.IMAGE_RECOLOR_DIR_PATH + "FINAL_RECOLORING_" + img_file_name
        self.recolor_left_img_file_path = self.IMAGE_RECOLOR_DIR_PATH + "LEFT_SIDE_RECOLOR_" + img_file_name
        self.recolor_right_img_file_path = self.IMAGE_RECOLOR_DIR_PATH + "RIGHT_SIDE_RECOLOR_" + img_file_name

        print("Image Location: ", self.original_img_file_path)

        # self.grayscaleMappings = {}
        # Setup Image Manipulation Variables
        self.original_image = Image.open(self.original_img_file_path)
        self.width, self.height = self.original_image.size
        self.image_edit = self.original_image.load()
        self.img_RGB_values = list(self.original_image.getdata())

        print("Image Dimensions: ", self.width, "x", self.height)

    def CREATE_RECOLORED_IMAGE_DIR(self):
        if not os.path.exists(self.TEST_FOLDER):  # creates /test-img-files/ if it does not exist
            os.mkdir(self.TEST_FOLDER)
        if not os.path.exists(self.IMAGE_DIR_PATH):  # creates subdir for image in /test-img-files/ if it does not exist
            os.mkdir(self.IMAGE_DIR_PATH)
        if not os.path.exists(self.IMAGE_RECOLOR_DIR_PATH):  # creates folder for agent if it does not exist
            os.mkdir(self.IMAGE_RECOLOR_DIR_PATH)

    def doesGrayscaleFileExist(self):
        return os.path.exists(self.grayscale_img_file_path)

    def grayscaleImageConversion(self):

        if not self.doesGrayscaleFileExist() or self.createGrayscaleRegardless:
            print("Converting image to grayscale...")
            for y in range(self.height):
                for x in range(self.width):
                    pixel = self.img_RGB_values[x + (self.width * y)]
                    GRAYSCALE_VALUE = RGB_TO_GRAY(pixel)

                    # try:
                    #     self.grayscaleMappings[GRAYSCALE_VALUE].append(pixel)
                    # except KeyError:
                    #     self.grayscaleMappings[GRAYSCALE_VALUE] = [pixel]

                    self.image_edit[x, y] = (GRAYSCALE_VALUE, GRAYSCALE_VALUE, GRAYSCALE_VALUE)
            self.original_image.save(self.grayscale_img_file_path)
            print("Grayscale conversion complete")
        else:
            print("Grayscale File Found")
            print("Image Variables Loaded")
