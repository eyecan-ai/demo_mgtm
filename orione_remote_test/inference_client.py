from orione.detections.rfs import PlanarOrientedRF
from orione.detections.commons import DetectionsDecoder
from orione.utils.filesystem import load_images_from_path
from orione.utils.drawing import PILKPLDrawer
import click
import imageio
from orione.interfaces.relate.client import RelateClient
import cv2
import rich
import sys


@click.command("Inference sample with Relate protocol")
@click.option("--host", required=True, help="Host")
@click.option("--port", default=44441, type=int, help="Port")
@click.option("--image_path", required=True, help="Path to target image")
@click.option("--min_score", default=0.25, help="Min predictions score")
def inference(host, port, image_path, min_score):

    # Create client
    client = RelateClient(host, port)

    # Load all images in folder
    images = load_images_from_path(image_path)

    # Images Generator
    images = list((imageio.imread(x) for x in images))

    # Iterate over images
    for image in images:

        # Inference
        output = client.inference([image])

        # Output image
        output_image = image.copy()

        # Checks for RFs in decoded output
        if DetectionsDecoder.OUTPUT_KEY_RFS in output:
            rfs = output[DetectionsDecoder.OUTPUT_KEY_RFS][0]

            output_image = PILKPLDrawer.draw_planar_rfs(
                output_image,
                [PlanarOrientedRF.encode(rf) for rf in rfs]
            )

            rich.print("[red] Reference Frames: [/red]")
            rich.print(rfs)

        cv2.imshow("image", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        if ord('q') == cv2.waitKey(0):
            sys.exit(0)


if __name__ == "__main__":
    inference()
