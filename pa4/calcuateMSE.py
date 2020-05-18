import numpy

class CalculateMSE:
    def __init__(self, image_RGB_values, image_predicted_RGB_values, width, height):

        self.image_RGB_values = numpy.array(image_RGB_values)
        self.image_predicted_RGB_values = numpy.array(image_predicted_RGB_values)

        self.image_RGB_values_left = None
        self.image_predicted_RGB_values_left = None

        self.image_RGB_values_right = None
        self.image_predicted_RGB_values_right = None
        self.width = width
        self.height = height

        self.left_side_MSE = None
        self.right_side_MSE = None

    def MSE(self):

        image_RGB_values_split = numpy.array_split(self.image_RGB_values,2)
        image_predicted_RGB_values_split = numpy.array_split(self.image_predicted_RGB_values,2)
        image_RGB_values_left = image_RGB_values_split[0]
        image_predicted_RGB_values_left = image_predicted_RGB_values_split[0]

        image_RGB_values_right = image_RGB_values_split[1]
        image_predicted_RGB_values_right = image_predicted_RGB_values_split[1]

        left_side_euclidean = numpy.linalg.norm(image_RGB_values_left - image_predicted_RGB_values_left, ord=2, axis=1) #spatial.distance.cdist(self.image_RGB_values_left, self.image_predicted_RGB_values_left)
        right_side_euclidean = numpy.linalg.norm(image_RGB_values_right - image_predicted_RGB_values_right, ord=2, axis=1)  #spatial.distance.cdist(self.image_RGB_values_right, self.image_predicted_RGB_values_right)

        left_side_sq = numpy.power(left_side_euclidean, 2)
        right_side_sq = numpy.power(right_side_euclidean, 2)

        self.left_side_MSE = numpy.mean(left_side_sq)
        self.right_side_MSE = numpy.mean(right_side_sq)



