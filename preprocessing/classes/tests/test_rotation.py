from labelstudio.LabelStudioInterface import LabelStudioInterface
from preprocessing.classes.AnnotatedPage import AnnotatedPage

LSI = LabelStudioInterface()


Ann = AnnotatedPage(LSI[367][1], img=LSI.get_image(367))

Ann.set_corrected_centroid()

img_boxes = [Ann.image_boxes[x_id] for x_id in Ann.image_boxes]

img_boxes.sort(key=lambda box: box.corrected_centroid[::-1])

for box in img_boxes:
    box.crop.show()
    input("Siguiente?")
