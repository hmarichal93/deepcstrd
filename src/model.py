import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize
import numpy as np
import segmentation_models_pytorch as smp
import cv2

from skimage import io, measure
from shapely.geometry import LineString
from pathlib import Path

from dataset import create_tiles_with_labels, from_tiles_to_image, overlay_images
from urudendro.image import load_image, write_image
from urudendro.drawing import Drawing, Color



class UNET:
    def __init__(self, weights_path = "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth" ,
                 tile_size=512, overlap=0.1, output_dir=None):
        self.model = self.load_model(weights_path)
        self.model.eval()
        self.tile_size = tile_size
        self.overlap = overlap
        self.output_dir = output_dir

    def load_model(self, weights_path):
        # Load your model here
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        model.load_state_dict(torch.load(weights_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        model = model.to(device)
        self.device = device
        return model

    def forward(self, img, output_dir=None):
        if output_dir:
            write_image(f"{output_dir}/img.png", img)
        image, _ = create_tiles_with_labels(img, img, tile_size=self.tile_size, overlap=self.overlap)
        image = torch.from_numpy(image)

        with torch.no_grad():
            #split the image into tiles
            image = image.permute(0, 3, 1, 2).float() / 255.0  # Normalize to [0, 1]
            # if self.device.type == "cuda":
            #     # convert rotated image to tensor
            #     image = image.to(self.device)
            pred = self.model(image)
            pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
            pred = pred.squeeze().cpu().numpy()  # Convert to numpy array

            pred = from_tiles_to_image(pred, self.tile_size, img, self.overlap, output_dir=output_dir, img=img)
            write_image(f"{output_dir}/pred.png", (pred * 255).astype(np.uint8))

        return pred


    def compute_connected_components_by_contour(self, skeleton, output_dir, debug=True):

        contours, hierarchy = cv2.findContours(skeleton.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if debug:
            from urudendro.drawing import Drawing, Color
            from shapely.geometry import LineString

            img = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
            color = Color()

        m_ch_e = []
        #sort contours by size descinding
        contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
        for idx, c in enumerate(contours):
            c_shape = c.shape[0]
            if c_shape < 2:
                continue

            m_ch_e.extend(c[:,0].tolist() + [[-1,-1]])

            if debug:
                poly = LineString(c[:, 0][:, ::-1])
                img = Drawing.curve(poly.coords, img, color.get_next_color())

        if debug:
            write_image(f"{output_dir}/contours.png", img)

        return np.array(m_ch_e)

    def compute_normals(self, m_ch_e, h,w):
        n = len(m_ch_e)
        Gy = np.zeros((h,w))
        Gx = np.zeros((h,w))

        for idx in range(n):
            p0 = m_ch_e[idx]
            p_m1 = m_ch_e[idx - 1] if idx > 0 else m_ch_e[idx]
            p_m2 = m_ch_e[idx + 1] if idx < n - 1 else m_ch_e[idx]

            if p0[0] == -1:
                continue

            tg = np.array(p_m2) - np.array(p_m1)
            tgnorm = tg / np.linalg.norm(tg)
            normal = np.array([-tgnorm[1], tgnorm[0]])

            i = int(p0[0])
            j = int(p0[1])
            Gy[j,i] = normal[0]
            Gx[j,i] = normal[1]


        return Gx, Gy




def rotate_image(image, center, angle=90):
    # Get image dimensions and calculate rotation matrix
    if angle == 0:
        return image
    (h, w) = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform rotation
    rotated_image = cv2.warpAffine(image.copy(), rotation_matrix, (w, h))
    return rotated_image
import cv2
import numpy as np

def rotate_image_with_padding(image, angle, cx, cy):
    h, w = image.shape[:2]

    # Calcular las distancias a las esquinas
    distances = [
        np.sqrt((cx - 0)**2 + (cy - 0)**2),
        np.sqrt((cx - w)**2 + (cy - 0)**2),
        np.sqrt((cx - 0)**2 + (cy - h)**2),
        np.sqrt((cx - w)**2 + (cy - h)**2)
    ]
    max_dist = int(np.ceil(max(distances))) // 2

    # Agregar padding
    padded_image = cv2.copyMakeBorder(
        image,
        max_dist, max_dist, max_dist, max_dist,
        borderType=cv2.BORDER_CONSTANT,
        value=255  # Color blanco
    )

    # Ajustar las coordenadas del centro
    new_cx, new_cy = cx + max_dist, cy + max_dist

    # Crear la matriz de rotación
    M = cv2.getRotationMatrix2D((new_cx, new_cy), angle, 1)

    # Rotar la imagen
    rotated_image = cv2.warpAffine(
        padded_image,
        M,
        (padded_image.shape[1], padded_image.shape[0]),
        borderValue=255
    )

    return rotated_image, max_dist

def unrotate_and_crop(image, angle, cx, cy, original_shape, max_dist):
    # Tamaño original
    original_h, original_w = original_shape

    # Ajustar el centro después del padding
    new_cx, new_cy = cx + max_dist, cy + max_dist

    # Anti-rotar la imagen
    M_inv = cv2.getRotationMatrix2D((new_cx, new_cy), -angle, 1)
    unrotated_image = cv2.warpAffine(
        image,
        M_inv,
        (image.shape[1], image.shape[0]),
        borderValue=255
    )

    # Recortar el padding
    cropped_image = unrotated_image[max_dist:max_dist+original_h, max_dist:max_dist+original_w]

    return cropped_image

def deep_learning_edge_detector(img,
                                weights_path= "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth",
                                output_dir=None, cy=None, cx=None, debug=False):
    model = UNET(weights_path)

    angle_range = [0,90,180,270]
    #angle_range = [0]
    pred_dict = {}
    #add padding to the image equal to the distance between the center and the border

    original_shape = img.shape[:2]
    for angle in angle_range:
        output_dir_angle = Path(output_dir) / f"{angle}"
        output_dir_angle.mkdir(parents=True, exist_ok=True)
        output_dir_angle = None if not debug else output_dir_angle
        rot_image = rotate_image(img, (cx, cy), angle=angle)
        #rot_image, max_dist = rotate_image_with_padding(img, angle, cx, cy)

        pred = model.forward(rot_image, output_dir=output_dir_angle)

        pred = rotate_image(pred, (cx, cy), angle=-angle)
        #pred = unrotate_and_crop(pred, angle, cx, cy, original_shape, max_dist)

        pred_dict[angle] = pred
    # Combine the predictions
    pred = np.zeros_like(pred_dict[angle])
    for angle in angle_range:
        pred += pred_dict[angle]

    #clip the values to 0
    # scale the values to 0-255
    if output_dir and debug:
        pred_scaled = (pred * 255).astype(np.uint8)
        #overlay the prediction on the image
        overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        overlay[:,:,0] = pred_scaled
        overlay = overlay_images(img.astype(np.uint8), overlay, alpha=0.5, beta=0.5, gamma=0)

        #draw over the image the center
        overlay = Drawing.circle(overlay, (cx, cy), thickness=-1, color=Color.red, radius=5)


        write_image(f"{output_dir}/overlay.png", overlay)



    #binarize the mask
    th = len(angle_range) / 3
    pred = (pred >= th).astype(np.uint8)  # Binarize the mask

    #skeletonizing the mask
    skeleton = skeletonize(pred)
    skeleton = np.where(skeleton, 255, 0)  # Skeletonize the mask

    # write the image to disk
    if output_dir and debug:
        write_image(f"{output_dir}/skel.png", skeleton)
    #m_ch_e = model.compute_connected_components(skeleton)
    m_ch_e = model.compute_connected_components_by_contour(skeleton, output_dir, debug)
    gx, gy = model.compute_normals(m_ch_e, img.shape[0], img.shape[1])
    if output_dir and debug:
        #write_image("labels.png", labels)
        debug_image = img.copy()

        amp = 10
        for idx in range(len(m_ch_e)):
            p0 = m_ch_e[idx]
            i = int(p0[0])
            j = int(p0[1])

            p1 = p0 + amp*np.array([gy[j,i], gx[j,i]])
            p1 = p1.astype(int)
            if p0[0] < 1 or p0[1] < 1 or p1[0] < 1 or p1[1] < 1:
                continue
            points = np.stack([p0,p1])[:,::-1]
            line = LineString(points)
            debug_image = Drawing.radii(line.coords, debug_image, color=Color.blue, thickness=1)

        for p in m_ch_e:
            if p[0] == -1:
                continue
            debug_image = Drawing.circle(debug_image, p, thickness=-1, color=Color.black, radius=1)

        write_image(f"{output_dir}/normals.png", debug_image)

    return m_ch_e, gy, gx
def test_forward(debug=False):
    weights_path = "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth"
    image_path = "/data/maestria/resultados/deep_cstrd/pinus_v1/val/images/segmented/F02d.png"
    img = load_image(image_path)
    #convert to torch tensor
    m_ch_e = deep_learning_edge_detector(img, weights_path)

    return m_ch_e



if __name__ == "__main__":
    test_forward()