import torch
import torchvision
from skimage.morphology import skeletonize
import cv2
import numpy as np
import segmentation_models_pytorch as smp

from shapely.geometry import LineString
from pathlib import Path
from urudendro.image import load_image, write_image
from urudendro.drawing import Drawing, Color

from deep_cstrd.dataset import create_tiles_with_labels, from_tiles_to_image, overlay_images, padding_image


class segmentation_model:
    UNET = 1
    UNET_PLUS_PLUS = 2
    MASK_RCNN = 3

class RingSegmentationModel:
    def __init__(self, weights_path = "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth" ,
                 tile_size=512, overlap=0.1, output_dir=None, model_type=segmentation_model.UNET):
        self.model_type = model_type
        self.model = self.load_model(weights_path)
        self.model.eval()
        self.tile_size = tile_size
        self.overlap = overlap
        self.output_dir = output_dir


    def load_model(self, weights_path, encoder='resnet18'):

        # Load your model here
        if self.model_type == segmentation_model.UNET:
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1
            )
        elif self.model_type == segmentation_model.UNET_PLUS_PLUS:
            model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1
            )
        elif self.model_type == segmentation_model.MASK_RCNN:
            model = torchvision.models.detection.mask_rcnn.MaskRCNN(backbone="resnet50", num_classes=1, pretrained=True)

        else:
            raise ValueError("Invalid model type")

        model.load_state_dict(torch.load(weights_path))
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        model = model.to(device)
        self.device = device
        return model

    def forward(self, img, output_dir=None, tile_size=0):
        if output_dir:
            write_image(f"{output_dir}/img.png", img)
        if tile_size>0:
            image, _ = create_tiles_with_labels(img, img, tile_size=self.tile_size, overlap=self.overlap)
        else:
            image = np.array([img])
        image = torch.from_numpy(image)

        with torch.no_grad():
            #split the image into tiles
            image = image.permute(0, 3, 1, 2).float() / 255.0  # Normalize to [0, 1]
            if self.device.type == "cuda":
                # convert rotated image to tensor
                image = image.to(self.device)
            pred = self.model(image)
            pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
            pred = pred.squeeze().cpu().numpy()  # Convert to numpy array
            if tile_size > 0:
                pred = from_tiles_to_image(pred, self.tile_size, img, self.overlap, output_dir=output_dir, img=img)

            if output_dir:
                write_image(f"{output_dir}/pred.png", (pred * 255).astype(np.uint8))

        return pred


    def compute_connected_components_by_contour(self, skeleton, output_dir, debug=True, minimum_length=10):

        contours, hierarchy = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if debug:
            from urudendro.drawing import Drawing, Color
            from shapely.geometry import LineString

            img = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
            color = Color()

        m_ch_e = []
        contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
        for idx, c in enumerate(contours):
            c = c.squeeze()
            c_shape = c.shape[0]
            if c_shape < minimum_length:
                continue

            m_ch_e.extend(c.tolist() + [[-1,-1]])

            if debug:
                poly = LineString(c[:, ::-1])
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

import numpy as np
from scipy.spatial import distance


def find_extreme_points(skeleton_points):
    """Encuentra los puntos extremos del esqueleto."""
    from scipy.spatial import cKDTree

    # Crear un árbol KD para calcular los vecinos
    tree = cKDTree(skeleton_points)
    extremes = []

    for point in skeleton_points:
        # Buscar vecinos dentro de un radio pequeño (8-conectividad)
        distances, indices = tree.query(point, k=17)
        neighbor_count = np.sum(distances < 1.5) - 1  # Ignorar el punto mismo
        if neighbor_count == 1:
            extremes.append(tuple(point))

    return extremes
def order_points_continuous(skeleton_points, start_point):
    """Ordena los puntos del esqueleto en una lista continua."""
    skeleton_points = set(map(tuple, skeleton_points))  # Convertir a tuplas para facilitar la búsqueda
    ordered_points = []
    current_point = start_point

    while skeleton_points:
        ordered_points.append(current_point)
        skeleton_points.remove(current_point)

        # Encontrar el vecino más cercano
        neighbors = [(current_point[0] + dx, current_point[1] + dy)
                     for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
        next_point = None
        for neighbor in neighbors:
            if neighbor in skeleton_points:
                next_point = neighbor
                break
        if next_point is None:
            break
        current_point = next_point

    return np.array(ordered_points)[:,[1,0]].tolist()

def regions_to_curves_with_skimage(skeleton):
    from skimage.measure import label, regionprops, find_contours
    contours = find_contours(skeleton, 0.5, fully_connected="high")
    m_ch_e = []
    for c in contours:
        c = np.round(c[:,[1,0]]).astype(int)
        m_ch_e.extend(c.tolist() + [[-1, -1]])
    return np.array(m_ch_e)

def regions_to_curves(skeleton):
    from skimage.measure import label, regionprops, find_contours

    labeled_image = label(skeleton, connectivity=1)
    m_ch_e = []
    for region_label in range(1, labeled_image.max() + 1):
        region_points = np.argwhere(labeled_image == region_label)
        extremes = find_extreme_points(region_points)
        if not extremes:
            continue
        ordered_points = order_points_continuous(region_points, extremes[0])
        if ordered_points is None:
            continue
        m_ch_e.extend(ordered_points + [[-1, -1]])
    m_ch_e = np.array(m_ch_e)
    return m_ch_e
def from_prediction_mask_to_curves(pred, model, output_dir=None, debug=False, ):
    from skimage.morphology import skeletonize
    skeleton = skeletonize(pred)
    #m_ch_e = regions_to_curves(skeleton)
    #m_ch_e = regions_to_curves_with_skimage(skeleton)
    m_ch_e = model.compute_connected_components_by_contour(np.where(skeleton, 255, 0), output_dir, debug)
    return m_ch_e

def deep_learning_edge_detector(img,
                                weights_path= "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth",
                                output_dir=None, cy=None, cx=None, debug=False, total_rotations=4, tile_size=0):

    h, w = img.shape[:2]
    if h % 32 != 0 or w % 32 != 0:
        img = padding_image(img, 32)

    model = RingSegmentationModel(weights_path)

    if total_rotations < 1:
        total_rotations = 1

    angle_range = np.arange(0,360, 360/total_rotations).tolist()

    pred_dict = {}
    for angle in angle_range:
        output_dir_angle = Path(output_dir) / f"{angle}"
        output_dir_angle.mkdir(parents=True, exist_ok=True)
        output_dir_angle = None if not debug else output_dir_angle

        rot_image = rotate_image(img, (cx, cy), angle=angle)
        pred = model.forward(rot_image, output_dir=output_dir_angle, tile_size=tile_size)
        pred = rotate_image(pred, (cx, cy), angle=-angle)
        pred_dict[angle] = pred

    # Combine the predictions computing the average
    pred = np.zeros_like(pred_dict[angle])
    for angle in angle_range:
        pred += pred_dict[angle]
    pred = pred / total_rotations
    #clip the values to 0
    # scale the values to 0-255
    if output_dir and debug:
        draw_pred_mask(pred, img, output_dir, cx, cy)

    #binarize the mask
    th = 0.5
    pred = (pred >= th).astype(np.uint8)  # Binarize the mask
    m_ch_e = from_prediction_mask_to_curves(pred, model, output_dir, debug)
    gx, gy = model.compute_normals(m_ch_e, img.shape[0], img.shape[1])
    if output_dir and debug:
        draw_normals(img, m_ch_e, gx, gy, output_dir)


    return m_ch_e, gy, gx

def draw_normals(img, m_ch_e, gx, gy, output_dir):
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
    return

def draw_pred_mask(pred, img, output_dir, cx, cy):
    pred_scaled = (pred * 255).astype(np.uint8)
    #overlay the prediction on the image
    overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    overlay[:,:,0] = pred_scaled
    overlay = overlay_images(img.astype(np.uint8), overlay, alpha=0.5, beta=0.5, gamma=0)

    #draw over the image the center
    overlay = Drawing.circle(overlay, (cx, cy), thickness=-1, color=Color.red, radius=5)


    write_image(f"{output_dir}/overlay.png", overlay)
    return
def test_forward(debug=False):
    weights_path = "/runs/unet_experiment/latest_model.pth"
    image_path = "/data/maestria/resultados/deep_cstrd/pinus_v1/val/images/segmented/F02d.png"
    img = load_image(image_path)
    #convert to torch tensor
    m_ch_e = deep_learning_edge_detector(img, weights_path)

    return m_ch_e



if __name__ == "__main__":
    test_forward()