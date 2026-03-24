from external_interfaces.LabelStudioInterface import LabelStudioInterface
from processing.AnnotatedPage import AnnotatedPage
from PIL import Image


def test_sindices(paths, lsi):

    for task in lsi.simplified_tasks[:50] + lsi.simplified_tasks[:-50]:
        img = Image.open(paths.get_image_path_from_task(task))

        for k, Ann in enumerate(
            (
                AnnotatedPage(ann, img, usernames_LS=lsi.usernames)
                for ann in task["annotations"]
            )
        ):

            sindices = [x.fragment.starting_index for x in Ann.image_boxes.values()]

            if not all(isinstance(x, int) for x in sindices):

                msg = f"No todos framgentos tienen asociado un int como starting_index: {[x.fragment.starting_index for x in Ann.image_boxes.values()]}. Son los siguientes:"

                for fragment in Ann.text_fragments.values():
                    if fragment.starting_index is None:
                        msg += "\n\t > " + fragment.text

                raise AssertionError(msg)

            first_of_paragraph = []
            for (
                paragraph
            ) in Ann.paragraphs:  # ahora comprobamos que vengan en orden ascendente
                first_of_paragraph.append(paragraph.text_fragments[0].starting_index)
                sindices_par = [f.starting_index for f in paragraph.text_fragments]

                assert (
                    -1 not in sindices_par
                )  # solamente puede haber un -1 si no está conectado a nada

                assert (
                    sorted(sindices_par) == sindices_par
                )  # a lo largo del código se asume que vienen así ordenados.
